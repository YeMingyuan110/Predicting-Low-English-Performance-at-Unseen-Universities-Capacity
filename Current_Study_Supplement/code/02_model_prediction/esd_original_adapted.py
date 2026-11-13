"""ESD-MTMS-Net 主模型。

本文件由旧版 ``models`` 包中的 tokenizer、attention、multiscale、Top-K FFN
和多任务头合并而成。论文主分析保持原始结构：理论分组多尺度注意力、动态
Top-K 激活和三个并行回归头；不加入旧探索阶段的 task adapter 或 linear skip。

特征顺序固定为 ``[连续特征, 类别特征]``。模型不负责拟合插补、标准化、
类别词表、PC1、风险阈值或 PLE 节点，这些操作必须由训练折内预处理完成。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AblationFlags:
    """只供正式消融实验使用；主模型采用默认值。"""

    attention: bool = True
    multiscale: bool = True
    topk: bool = True
    dense: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """旧版正式配置的默认容量；折内寻优可覆盖数值参数。"""

    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 8
    ff_ratio: int = 4
    dropout: float = 0.12
    topk_ratio: float = 0.50
    topk_temperature: float = 0.75
    head_hidden: int = 128
    variant: str = "v1"
    soft_threshold_bins: int = 8
    soft_threshold_temperature: float = 0.50
    num_task_experts: int = 4


class FeatureTokenizer(nn.Module):
    """将连续和类别字段分别编码为特征 token，并添加 CLS token。"""

    def __init__(
        self,
        num_continuous: int,
        categorical_cardinalities: Sequence[int],
        d_model: int,
        dropout: float,
        soft_threshold_bins: int = 0,
        soft_threshold_temperature: float = 0.50,
    ) -> None:
        super().__init__()
        cards = tuple(int(value) for value in categorical_cardinalities)
        if num_continuous < 0 or any(value <= 0 for value in cards):
            raise ValueError("连续特征数不能为负，类别基数必须为正")
        if num_continuous + len(cards) == 0:
            raise ValueError("模型至少需要一个输入特征")
        self.num_continuous = int(num_continuous)
        self.cardinalities = cards
        self.d_model = int(d_model)
        self.soft_threshold_bins = int(soft_threshold_bins)
        if self.soft_threshold_bins < 0:
            raise ValueError("软阈值基函数数量不能为负")
        if soft_threshold_temperature <= 0.0:
            raise ValueError("软阈值温度必须为正")
        self.continuous_weight = nn.Parameter(torch.empty(num_continuous, d_model))
        self.continuous_bias = nn.Parameter(torch.empty(num_continuous, d_model))
        if self.soft_threshold_bins:
            initial_thresholds = torch.linspace(
                -2.5, 2.5, steps=self.soft_threshold_bins
            ).repeat(num_continuous, 1)
            self.soft_thresholds = nn.Parameter(initial_thresholds)
            self.soft_threshold_projection = nn.Parameter(
                torch.empty(num_continuous, self.soft_threshold_bins, d_model)
            )
            self.log_soft_threshold_temperature = nn.Parameter(
                torch.full(
                    (num_continuous, 1),
                    math.log(math.expm1(float(soft_threshold_temperature))),
                )
            )
        else:
            self.register_parameter("soft_thresholds", None)
            self.register_parameter("soft_threshold_projection", None)
            self.register_parameter("log_soft_threshold_temperature", None)
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, d_model, padding_idx=0) for cardinality in cards]
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.feature_identity = nn.Embedding(self.num_features + 1, d_model)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    @property
    def num_categorical(self) -> int:
        return len(self.cardinalities)

    @property
    def num_features(self) -> int:
        return self.num_continuous + self.num_categorical

    def reset_parameters(self) -> None:
        nn.init.normal_(self.continuous_weight, std=0.02)
        nn.init.zeros_(self.continuous_bias)
        if self.soft_threshold_projection is not None:
            nn.init.normal_(self.soft_threshold_projection, std=0.01)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.feature_identity.weight, std=0.02)
        for embedding in self.category_embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()

    def forward(
        self,
        continuous: Tensor,
        categorical: Tensor,
        feature_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch = continuous.shape[0]
        if continuous.shape != (batch, self.num_continuous):
            raise ValueError("连续特征矩阵形状与模型定义不一致")
        if categorical.shape != (batch, self.num_categorical):
            raise ValueError("类别特征矩阵形状与模型定义不一致")
        if feature_mask is None:
            observed = torch.ones(
                batch, self.num_features, dtype=torch.bool, device=continuous.device
            )
        else:
            if feature_mask.shape != (batch, self.num_features):
                raise ValueError("feature_mask 形状不正确")
            observed = feature_mask.to(device=continuous.device, dtype=torch.bool)

        values = torch.where(observed[:, : self.num_continuous], continuous, 0.0)
        numeric_values = values.to(self.continuous_weight.dtype)
        continuous_tokens = (
            numeric_values.unsqueeze(-1) * self.continuous_weight.unsqueeze(0)
            + self.continuous_bias.unsqueeze(0)
        )
        if self.soft_thresholds is not None:
            assert self.soft_threshold_projection is not None
            assert self.log_soft_threshold_temperature is not None
            temperature = F.softplus(self.log_soft_threshold_temperature).clamp_min(0.05)
            threshold_response = torch.sigmoid(
                (numeric_values.unsqueeze(-1) - self.soft_thresholds.unsqueeze(0))
                / temperature.unsqueeze(0)
            )
            continuous_tokens = continuous_tokens + torch.einsum(
                "bfk,fkd->bfd", threshold_response, self.soft_threshold_projection
            )
        pieces = [continuous_tokens]
        category_observed = observed[:, self.num_continuous :]
        category_tokens = []
        for index, (embedding, cardinality) in enumerate(
            zip(self.category_embeddings, self.cardinalities)
        ):
            codes = categorical[:, index].long()
            valid_codes = (codes >= 0) & (codes < cardinality)
            # 类别编码已在折内预处理阶段校验。CUDA上的逐列 ``bool`` 检查会在
            # 每个batch强制CPU/GPU同步，因此只为直接CPU调用保留防御性检查。
            if not codes.is_cuda and not bool(
                torch.all(valid_codes | ~category_observed[:, index])
            ):
                raise ValueError(f"第 {index} 个类别特征存在越界编码")
            shifted = torch.where(category_observed[:, index], codes + 1, 0)
            category_tokens.append(embedding(shifted))
        if category_tokens:
            pieces.append(torch.stack(category_tokens, dim=1))

        feature_tokens = torch.cat(pieces, dim=1)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, feature_tokens], dim=1)
        identity = torch.arange(self.num_features + 1, device=tokens.device)
        tokens = self.dropout(tokens + self.feature_identity(identity).unsqueeze(0))
        token_mask = torch.cat(
            [torch.ones(batch, 1, dtype=torch.bool, device=tokens.device), observed], dim=1
        )
        return tokens * token_mask.unsqueeze(-1), token_mask


class GlobalFeatureAttention(nn.Module):
    """特征自注意力与 CLS 全局查询的双通路融合。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.feature_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.global_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.feature_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.global_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        padding_mask = ~token_mask.bool()
        feature_output, _ = self.feature_attention(
            tokens, tokens, tokens, key_padding_mask=padding_mask, need_weights=False
        )
        global_output, _ = self.global_attention(
            tokens[:, :1], tokens, tokens, key_padding_mask=padding_mask, need_weights=False
        )
        feature_output = self.feature_gate(tokens) * feature_output
        global_output = self.global_gate(tokens[:, :1]) * global_output
        cls_only = torch.cat([global_output, torch.zeros_like(tokens[:, 1:])], dim=1)
        return (feature_output + cls_only) * token_mask.unsqueeze(-1)


class TheoryGroupPool(nn.Module):
    """对一个预先定义的理论特征组执行注意力池化。"""

    def __init__(self, indices: Sequence[int], d_model: int) -> None:
        super().__init__()
        self.register_buffer("indices", torch.tensor(tuple(indices), dtype=torch.long))
        self.query = nn.Parameter(torch.empty(d_model))
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        nn.init.normal_(self.query, std=0.02)

    def forward(self, features: Tensor, feature_mask: Tensor) -> tuple[Tensor, Tensor]:
        members = features.index_select(1, self.indices)
        valid = feature_mask.index_select(1, self.indices)
        scores = torch.einsum("bnd,d->bn", members, self.query) / math.sqrt(members.shape[-1])
        scores = scores.masked_fill(~valid, -torch.finfo(scores.dtype).max)
        weights = torch.softmax(scores, dim=-1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        context = self.projection(torch.einsum("bn,bnd->bd", weights, members))
        return context * valid.any(dim=-1, keepdim=True), valid


class TheoryGroupedMultiScale(nn.Module):
    """按教育理论组聚合局部尺度，再与 CLS 全局尺度融合。"""

    def __init__(
        self,
        d_model: int,
        num_features: int,
        theory_groups: Mapping[str, Sequence[int]],
    ) -> None:
        super().__init__()
        if not theory_groups:
            raise ValueError("启用多尺度模块时 theory_groups 不能为空")
        checked: list[tuple[int, ...]] = []
        for name, raw_indices in theory_groups.items():
            indices = tuple(int(value) for value in raw_indices)
            if not indices or min(indices) < 0 or max(indices) >= num_features:
                raise ValueError(f"理论组 {name!r} 的特征索引无效")
            if len(indices) != len(set(indices)):
                raise ValueError(f"理论组 {name!r} 含重复索引")
            checked.append(indices)
        self.groups = nn.ModuleList([TheoryGroupPool(values, d_model) for values in checked])
        self.cls_query = nn.Linear(d_model, d_model, bias=False)
        self.cls_projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        features, feature_mask = tokens[:, 1:], token_mask[:, 1:]
        # CUDA autocast下输入token可能仍为Float，而线性层产生Half/BFloat16。
        # 延迟到首个理论组输出后再创建累加区，保证index_add_两端dtype一致。
        member_updates: Tensor | None = None
        member_counts: Tensor | None = None
        contexts, group_valid = [], []
        for group in self.groups:
            context, valid = group(features, feature_mask)
            if member_updates is None:
                member_updates = torch.zeros(
                    features.shape, dtype=context.dtype, device=features.device
                )
                member_counts = torch.zeros(
                    *features.shape[:2], 1,
                    dtype=context.dtype,
                    device=features.device,
                )
            contexts.append(context)
            group_valid.append(valid.any(dim=-1))
            expanded = context.unsqueeze(1).expand(-1, group.indices.numel(), -1)
            selected = valid.unsqueeze(-1).to(context.dtype)
            member_updates.index_add_(1, group.indices, expanded * selected)
            assert member_counts is not None
            member_counts.index_add_(1, group.indices, selected)
        assert member_updates is not None and member_counts is not None
        feature_delta = member_updates / member_counts.clamp_min(1.0)

        context_matrix = torch.stack(contexts, dim=1)
        valid_matrix = torch.stack(group_valid, dim=1)
        scores = torch.einsum(
            "bd,bgd->bg", self.cls_query(tokens[:, 0]), context_matrix
        ) / math.sqrt(tokens.shape[-1])
        scores = scores.masked_fill(~valid_matrix, -torch.finfo(scores.dtype).max)
        weights = torch.softmax(scores, dim=-1) * valid_matrix.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cls_delta = self.cls_projection(torch.einsum("bg,bgd->bd", weights, context_matrix))
        return torch.cat([cls_delta.unsqueeze(1), feature_delta], dim=1) * token_mask.unsqueeze(-1)


class DenseFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class AdaptiveTopKFFN(nn.Module):
    """硬 Top-K 前向掩码与软直通梯度；表示稀疏性而非虚报算力节省。"""

    def __init__(
        self, d_model: int, hidden: int, topk_ratio: float, temperature: float, dropout: float
    ) -> None:
        super().__init__()
        if not 0.0 < topk_ratio <= 1.0 or temperature <= 0.0:
            raise ValueError("topk_ratio 须位于 (0,1]，temperature 须为正")
        self.active_units = max(1, math.ceil(hidden * topk_ratio))
        self.temperature = float(temperature)
        self.value = nn.Linear(d_model, hidden)
        self.router = nn.Linear(d_model, hidden)
        self.output = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        logits = self.router(inputs)
        indices = torch.topk(logits, self.active_units, dim=-1, sorted=False).indices
        hard = torch.zeros_like(logits).scatter(-1, indices, 1.0)
        soft = torch.softmax(logits / self.temperature, dim=-1) * self.active_units
        mask = hard + soft - soft.detach()
        return self.dropout(self.output(self.dropout(F.gelu(self.value(inputs))) * mask))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        num_features: int,
        theory_groups: Mapping[str, Sequence[int]],
        flags: AblationFlags,
    ) -> None:
        super().__init__()
        hidden = config.d_model * config.ff_ratio
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.multiscale_norm = nn.LayerNorm(config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.attention = (
            GlobalFeatureAttention(config.d_model, config.num_heads, config.dropout)
            if flags.attention else None
        )
        self.multiscale = (
            TheoryGroupedMultiScale(config.d_model, num_features, theory_groups)
            if flags.multiscale else None
        )
        self.ffn: nn.Module | None
        if flags.topk:
            self.ffn = AdaptiveTopKFFN(
                config.d_model, hidden, config.topk_ratio,
                config.topk_temperature, config.dropout,
            )
        elif flags.dense:
            self.ffn = DenseFFN(config.d_model, hidden, config.dropout)
        else:
            self.ffn = None

    @staticmethod
    def _residual(tokens: Tensor, delta: Tensor, mask: Tensor, dropout: nn.Module) -> Tensor:
        return (tokens + dropout(delta)) * mask.unsqueeze(-1)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        if self.attention is not None:
            tokens = self._residual(
                tokens, self.attention(self.attention_norm(tokens), mask), mask,
                self.residual_dropout,
            )
        if self.multiscale is not None:
            tokens = self._residual(
                tokens, self.multiscale(self.multiscale_norm(tokens), mask), mask,
                self.residual_dropout,
            )
        if self.ffn is not None:
            tokens = self._residual(
                tokens, self.ffn(self.ffn_norm(tokens)), mask, self.residual_dropout
            )
        return tokens


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs).squeeze(-1)


class GatedTaskExperts(nn.Module):
    """任务门控专家头。

    各任务在共享表征上选择不同专家，用于减少弱相关工程任务
    对英语任务的负迁移。该模块不接收树模型预测。
    """

    def __init__(
        self, d_model: int, hidden: int, dropout: float, num_tasks: int, num_experts: int
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("任务门控专家数至少为2")
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_experts)
            ]
        )
        self.gates = nn.ModuleList(
            [nn.Linear(d_model, num_experts) for _ in range(num_tasks)]
        )
        self.towers = nn.ModuleList(
            [RegressionHead(d_model, hidden, dropout) for _ in range(num_tasks)]
        )

    def forward(self, shared: Tensor) -> Tensor:
        expert_values = torch.stack([expert(shared) for expert in self.experts], dim=1)
        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            weights = torch.softmax(gate(shared), dim=-1)
            task_value = torch.einsum("be,bed->bd", weights, expert_values)
            outputs.append(tower(shared + task_value))
        return torch.stack(outputs, dim=-1)


class ESDMTMSNet(nn.Module):
    """固定结构的 ESD-MTMS-Net，输出三个并行连续任务。"""

    def __init__(
        self,
        num_continuous: int,
        categorical_cardinalities: Sequence[int],
        theory_groups: Mapping[str, Sequence[int]],
        task_names: Sequence[str] = (
            "english_performance",
            "non_english_generic_pc1",
            "engineering_project_formulation",
        ),
        config: ModelConfig | None = None,
        flags: AblationFlags | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.flags = flags or AblationFlags()
        if self.config.variant not in {"v1", "v2"}:
            raise ValueError("模型版本只能是 v1 或 v2")
        if self.config.d_model % self.config.num_heads:
            raise ValueError("d_model 必须能被 num_heads 整除")
        if self.flags.topk and self.flags.dense:
            raise ValueError("同一实验不能同时启用 Top-K FFN 与 Dense FFN")
        self.task_names = tuple(task_names)
        if not self.task_names or len(set(self.task_names)) != len(self.task_names):
            raise ValueError("任务名称必须非空且互不重复")
        self.tokenizer = FeatureTokenizer(
            num_continuous, categorical_cardinalities,
            self.config.d_model, self.config.dropout,
            soft_threshold_bins=(
                self.config.soft_threshold_bins if self.config.variant == "v2" else 0
            ),
            soft_threshold_temperature=self.config.soft_threshold_temperature,
        )
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(self.config, self.tokenizer.num_features, theory_groups, self.flags)
                for _ in range(self.config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.shared_projection = nn.Sequential(
            nn.Linear(self.config.d_model * 2, self.config.d_model),
            nn.GELU(),
            nn.LayerNorm(self.config.d_model),
        )
        if self.config.variant == "v2":
            self.task_experts: GatedTaskExperts | None = GatedTaskExperts(
                self.config.d_model,
                self.config.head_hidden,
                self.config.dropout,
                len(self.task_names),
                self.config.num_task_experts,
            )
            self.heads: nn.ModuleList | None = None
        else:
            self.task_experts = None
            self.heads = nn.ModuleList(
                [
                    RegressionHead(
                        self.config.d_model, self.config.head_hidden, self.config.dropout
                    )
                    for _ in self.task_names
                ]
            )

    def encode(
        self, continuous: Tensor, categorical: Tensor, feature_mask: Tensor | None = None
    ) -> Tensor:
        tokens, mask = self.tokenizer(continuous, categorical, feature_mask)
        for block in self.blocks:
            tokens = block(tokens, mask)
        tokens = self.final_norm(tokens) * mask.unsqueeze(-1)
        feature_mask_float = mask[:, 1:].unsqueeze(-1).to(tokens.dtype)
        feature_mean = (tokens[:, 1:] * feature_mask_float).sum(dim=1)
        feature_mean = feature_mean / feature_mask_float.sum(dim=1).clamp_min(1.0)
        return self.shared_projection(torch.cat([tokens[:, 0], feature_mean], dim=-1))

    def forward(
        self, continuous: Tensor, categorical: Tensor, feature_mask: Tensor | None = None
    ) -> Tensor:
        shared = self.encode(continuous, categorical, feature_mask)
        if self.task_experts is not None:
            return self.task_experts(shared)
        assert self.heads is not None
        return torch.stack([head(shared) for head in self.heads], dim=-1)


def build_model(
    num_continuous: int,
    categorical_cardinalities: Sequence[int],
    theory_groups: Mapping[str, Sequence[int]],
    *,
    config: ModelConfig | None = None,
    flags: AblationFlags | None = None,
) -> ESDMTMSNet:
    """训练程序唯一应调用的模型工厂。"""

    return ESDMTMSNet(
        num_continuous=num_continuous,
        categorical_cardinalities=categorical_cardinalities,
        theory_groups=theory_groups,
        config=config,
        flags=flags,
    )


def build_english_model(
    num_continuous: int,
    categorical_cardinalities: Sequence[int],
    theory_groups: Mapping[str, Sequence[int]],
    *, mode: str = "M1", config: ModelConfig | None = None,
    flags: AblationFlags | None = None,
) -> ESDMTMSNet:
    """新实验：M0为英语成绩；M1增加低表现风险logit，概率需单独校准。

    输出列0为标准化英语成绩，M1输出列1为未校准logit。T0只传实际
    非空理论组；类别索引遵循FeatureTokenizer文档，不应传独热展开列。
    """
    if mode not in {"M0", "M1"}:
        raise ValueError("mode必须为M0或M1")
    groups = {name: list(indices) for name, indices in theory_groups.items() if indices}
    names = ("english_performance",) if mode == "M0" else ("english_performance", "low_english_risk_logit")
    return ESDMTMSNet(num_continuous, categorical_cardinalities, groups,
                      task_names=names, config=config, flags=flags)


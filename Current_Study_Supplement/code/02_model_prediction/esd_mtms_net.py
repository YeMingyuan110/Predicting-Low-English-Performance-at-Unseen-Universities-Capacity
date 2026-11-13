"""ESD-MTMS-Net M0/M1：面向英语连续成绩与低表现风险的表格网络。

M0仅预测连续英语成绩；M1在同一表征上联合预测连续成绩与低表现风险。
模型改进候选（门控残差、理论分组聚合、稀疏Top-K前馈）只能在内层高校验证中
选择，绝不能根据外层测试折结果决定是否保留。该文件不宣称模型必然优于基线。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ESDConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.15
    topk_ratio: float = 0.50
    batch_size: int = 1024
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    max_epochs: int = 120
    patience: int = 15
    risk_weight: float = 0.35
    rank_weight: float = 0.05
    grad_clip: float = 1.0
    categorical_cardinalities: tuple[int, ...] = ()
    backend: str = "torch"
    mixed_precision: bool = True


class OriginalArchitectureAdapter(nn.Module):
    """连接完整旧架构与新实验训练接口；最后若干列是类别索引而非连续值。"""
    def __init__(self, n_features, groups, cfg, mode="M1"):
        super().__init__()
        from esd_original_adapted import ModelConfig, build_english_model
        self.cfg, self.mode = cfg, mode
        self.n_numeric = n_features - len(cfg.categorical_cardinalities)
        self.core = build_english_model(
            self.n_numeric, cfg.categorical_cardinalities,
            {str(i): list(g) for i, g in enumerate(groups) if len(g)}, mode=mode,
            config=ModelConfig(d_model=cfg.d_model, num_heads=cfg.n_heads,
                num_layers=cfg.n_layers, dropout=cfg.dropout,
                topk_ratio=cfg.topk_ratio, head_hidden=cfg.d_model))

    def forward(self, x):
        raw = self.core(x[:, :self.n_numeric], x[:, self.n_numeric:].long())
        out = {"english_score": raw[:, 0]}
        if self.mode == "M1":
            out["risk_logit"] = raw[:, 1]
        return out


# 所有训练/重训函数统一使用完整旧网络。
ESDMTMSNet = OriginalArchitectureAdapter


def _seed_everything(seed):
    """固定CPU/CUDA和批次顺序；记录环境以解释跨硬件数值差异。"""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def _gpu_enabled(device, cfg):
    return device.type == "cuda" and cfg.mixed_precision


def _epoch_batch_factory(x, y, risk, device, batch_size, seed):
    """Keep this small tabular dataset on CUDA and shuffle indices there each epoch."""
    if device.type == "cuda":
        tx=torch.as_tensor(x,dtype=torch.float32,device=device)
        ty=torch.as_tensor(y,dtype=torch.float32,device=device)
        tr=torch.as_tensor(risk,dtype=torch.float32,device=device)
        generator=torch.Generator(device=device).manual_seed(seed)
        def batches():
            order=torch.randperm(len(tx),device=device,generator=generator)
            for indices in order.split(batch_size):
                yield tx[indices],ty[indices],tr[indices]
        return batches
    loader=DataLoader(TensorDataset(torch.tensor(x).float(),torch.tensor(y).float(),
        torch.tensor(risk).float()),batch_size=batch_size,shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    return lambda:iter(loader)


def _validation_metric(model, x_valid, y_valid, risk_valid, mode):
    """Evaluate the same RMSE/Brier endpoint with one host synchronization per epoch."""
    device=next(model.parameters()).device
    xv=torch.as_tensor(x_valid,dtype=torch.float32,device=device)
    yv=torch.as_tensor(y_valid,dtype=torch.float32,device=device)
    rv=torch.as_tensor(risk_valid,dtype=torch.float32,device=device)
    model.eval()
    total=torch.zeros((),device=device,dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0,len(xv),model.cfg.batch_size):
            with torch.autocast(device_type="cuda",dtype=torch.float16,
                                enabled=device.type=="cuda" and model.cfg.mixed_precision):
                out=model(xv[start:start+model.cfg.batch_size])
                errors=(torch.sigmoid(out["risk_logit"])-rv[start:start+model.cfg.batch_size]
                        if mode=="M1" else out["english_score"]-yv[start:start+model.cfg.batch_size])
            total+=errors.double().square().sum()
    mse=(total/len(xv)).item()
    return mse if mode=="M1" else mse**.5


def _loss(model, xb, yb, rb):
    out=model(xb)
    loss=torch.nn.functional.mse_loss(out["english_score"],yb)
    if model.cfg.rank_weight:
        loss=loss+model.cfg.rank_weight*_pairwise_loss(out["english_score"],yb)
    if model.mode=="M1":
        loss=loss+model.cfg.risk_weight*torch.nn.functional.binary_cross_entropy_with_logits(out["risk_logit"],rb)
    if not torch.isfinite(loss): raise FloatingPointError("训练损失非有限")
    return loss


def _pairwise_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    dp, dy = pred[:, None] - pred[None, :], target[:, None] - target[None, :]
    mask = dy.abs() > 1e-7
    # Avoid a CPU/GPU synchronization and a dynamic-size masked copy each batch.
    terms=torch.nn.functional.softplus(-torch.sign(dy)*dp)
    return (terms*mask).sum()/mask.sum().clamp_min(1)


def fit_esd(
    x_train: np.ndarray, y_train: np.ndarray, risk_train: np.ndarray,
    x_valid: np.ndarray, y_valid: np.ndarray, risk_valid: np.ndarray,
    groups: Sequence[Sequence[int]], cfg: ESDConfig, mode: str, seed: int,
) -> tuple[ESDMTMSNet, dict]:
    """使用内层验证早停；返回最佳权重和可审计训练摘要。"""
    if cfg.backend=="lightning":
        return _lightning_fit(x_train,y_train,risk_train,groups,cfg,mode,seed,
                              validation=(x_valid,y_valid,risk_valid))
    if cfg.backend!="torch": raise ValueError("未知训练后端")
    _seed_everything(seed)
    if cfg.max_epochs<1 or cfg.patience<1: raise ValueError("训练轮次和早停耐心值必须为正")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESDMTMSNet(x_train.shape[1], groups, cfg, mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    gpu=_gpu_enabled(device,cfg)
    epoch_batches=_epoch_batch_factory(x_train,y_train,risk_train,device,cfg.batch_size,seed)
    validation=(torch.as_tensor(x_valid,dtype=torch.float32,device=device),
                torch.as_tensor(y_valid,dtype=torch.float32,device=device),
                torch.as_tensor(risk_valid,dtype=torch.float32,device=device))
    scaler=torch.amp.GradScaler("cuda",enabled=gpu)
    best, best_loss, stale, best_epoch = None, float("inf"), 0, 0
    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for xb, yb, rb in epoch_batches():
            xb, yb, rb = xb.to(device,non_blocking=gpu), yb.to(device,non_blocking=gpu), rb.to(device,non_blocking=gpu)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=gpu):
                loss = _loss(model, xb, yb, rb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
        val_loss=_validation_metric(model,*validation,mode)
        if not np.isfinite(val_loss): raise FloatingPointError("验证指标非有限")
        history.append({"epoch": epoch, "validation_score": val_loss,"metric":"Brier" if mode=="M1" else "RMSE_standardized"})
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best)
    return model, {"best_epoch": best_epoch, "best_validation_score": best_loss,
                   "selection_metric":"Brier" if mode=="M1" else "RMSE_standardized",
                   "config": asdict(cfg), "history": history, "device": str(device)}


def predict_esd(model: ESDMTMSNet, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        predictions=[]; probabilities=[]
        for start in range(0,len(x),model.cfg.batch_size):
            xb=torch.tensor(x[start:start+model.cfg.batch_size]).float().to(device,non_blocking=device.type=="cuda")
            with torch.autocast(device_type="cuda",dtype=torch.float16,
                                enabled=device.type=="cuda" and model.cfg.mixed_precision):
                out=model(xb)
            predictions.append(out["english_score"].cpu().numpy())
            if "risk_logit" in out: probabilities.append(torch.sigmoid(out["risk_logit"]).cpu().numpy())
        score=np.concatenate(predictions)
        risk=np.concatenate(probabilities) if probabilities else None
    return score, risk


def train_esd_fixed(
    x: np.ndarray, y: np.ndarray, risk: np.ndarray, groups: Sequence[Sequence[int]],
    cfg: ESDConfig, mode: str, seed: int, epochs: int,
) -> ESDMTMSNet:
    """用内层验证预先选定的轮次在完整外层训练集重训，不接触外层测试标签。"""
    if cfg.backend=="lightning":
        return _lightning_fit(x,y,risk,groups,cfg,mode,seed,epochs=epochs)[0]
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESDMTMSNet(x.shape[1], groups, cfg, mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    gpu=_gpu_enabled(device,cfg)
    epoch_batches=_epoch_batch_factory(x,y,risk,device,cfg.batch_size,seed)
    scaler=torch.amp.GradScaler("cuda",enabled=gpu)
    for _ in range(max(1, int(epochs))):
        model.train()
        for xb, yb, rb in epoch_batches():
            xb, yb, rb = xb.to(device,non_blocking=gpu), yb.to(device,non_blocking=gpu), rb.to(device,non_blocking=gpu)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=gpu):
                loss = _loss(model, xb, yb, rb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
    return model


def _lightning_fit(x,y,risk,groups,cfg,mode,seed,validation=None,epochs=None):
    """可选Lightning训练后端；数据划分由调用方完成，不随机拆分高校。

    最优权重保存在内存；禁止logger/checkpoint默认向项目目录写入过程文件。
    正式重训固定内层选出的轮次，不再早停。默认仍使用原生torch后端。
    """
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl
    _seed_everything(seed)
    class Module(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.network=ESDMTMSNet(x.shape[1],groups,cfg,mode)
            self.history=[]; self.best_state=None; self.best=float("inf")
            self.best_epoch=0; self.stale=0
        def training_step(self,batch,batch_idx):
            return _loss(self.network,*batch)
        def on_validation_epoch_start(self):
            self.error_sum=0.; self.validation_n=0
        def validation_step(self,batch,batch_idx):
            xb,yb,rb=batch; out=self.network(xb)
            errors=((torch.sigmoid(out["risk_logit"])-rb)**2 if mode=="M1"
                    else (out["english_score"]-yb)**2)
            self.error_sum+=errors.detach().sum().item(); self.validation_n+=len(yb)
        def on_validation_epoch_end(self):
            value=self.error_sum/self.validation_n
            if mode=="M0": value=value**.5
            if not np.isfinite(value): raise FloatingPointError("Lightning验证指标非有限")
            epoch=self.current_epoch+1
            self.history.append({"epoch":epoch,"validation_score":value})
            if value<self.best-1e-5:
                self.best=value; self.best_epoch=epoch; self.stale=0
                self.best_state={k:v.detach().cpu().clone() for k,v in self.network.state_dict().items()}
            else: self.stale+=1
            if self.stale>=cfg.patience: self.trainer.should_stop=True
        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    def loader(values,shuffle):
        return DataLoader(TensorDataset(*(torch.as_tensor(a,dtype=torch.float32) for a in values)),
            batch_size=cfg.batch_size,shuffle=shuffle,num_workers=0,
            generator=torch.Generator().manual_seed(seed))
    module=Module()
    trainer=pl.Trainer(max_epochs=cfg.max_epochs if validation is not None else max(1,int(epochs)),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",devices=1,
        logger=False,enable_checkpointing=False,enable_model_summary=False,
        enable_progress_bar=False,num_sanity_val_steps=0,gradient_clip_val=cfg.grad_clip,
        deterministic=True,precision="16-mixed" if cfg.mixed_precision and torch.cuda.is_available() else "32-true")
    trainer.fit(module,loader((x,y,risk),True),
                val_dataloaders=loader(validation,False) if validation is not None else None)
    if validation is not None:
        if module.best_state is None: raise RuntimeError("Lightning没有产生有效验证结果")
        module.network.load_state_dict(module.best_state)
    return module.network,{"best_epoch":module.best_epoch if validation is not None else epochs,
        "best_validation_score":module.best,"selection_metric":"Brier" if mode=="M1" else "RMSE_standardized",
        "config":asdict(cfg),"history":module.history,"backend":"lightning"}

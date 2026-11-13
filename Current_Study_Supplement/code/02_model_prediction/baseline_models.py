"""同协议经典与近期表格基线模型注册表。

模型只定义在这里；外层/内层高校隔离、阈值和OOF预测由
``validation_and_tuning.py``统一控制。可选依赖缺失时明确报为不可用，不允许
静默用另一模型替换后仍沿用原名称。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import os

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np


class _TabMBase(BaseEstimator):
    """官方TabM适配器：训练成员损失均值，推断平均成员预测。

    固定训练轮次作为内层搜索参数，不在学生随机留出集早停。预处理仍由
    外层Pipeline在当前训练高校拟合。此版本使用基础TabM，无额外数值嵌入。
    """
    def __init__(self, width=128, depth=2, dropout=.1, learning_rate=.002,
                 epochs=80, batch_size=1024, seed=42, k=32,
                 mixed_precision=True):
        self.width=width; self.depth=depth; self.dropout=dropout
        self.learning_rate=learning_rate; self.epochs=epochs
        self.batch_size=batch_size; self.seed=seed; self.k=k
        self.mixed_precision=mixed_precision

    def fit(self, X, y):
        import torch
        from tabm import TabM
        from torch.utils.data import TensorDataset, DataLoader
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.set_float32_matmul_precision("high")
        x=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.float32)
        self.n_features_in_=x.shape[1]
        self.device_="cuda" if torch.cuda.is_available() else "cpu"
        classification=isinstance(self,ClassifierMixin)
        if classification:
            self.classes_=np.unique(y)
            if not np.array_equal(self.classes_,[0,1]): raise ValueError("TabM分类需要0/1两类")
            self.mean_,self.scale_=0.,1.
        else: self.mean_,self.scale_=float(y.mean()),max(float(y.std()),1e-8)
        target=(y-self.mean_)/self.scale_
        self.model_=TabM.make(n_num_features=x.shape[1],d_out=1,k=self.k,
            n_blocks=self.depth,d_block=self.width,dropout=self.dropout).to(self.device_)
        optimizer=torch.optim.AdamW(self.model_.parameters(),lr=self.learning_rate,weight_decay=.0003)
        gpu=self.device_=="cuda"
        if gpu:
            x_device=torch.as_tensor(x,device=self.device_)
            y_device=torch.as_tensor(target,device=self.device_)
            generator=torch.Generator(device=self.device_).manual_seed(self.seed)
            def batches():
                order=torch.randperm(len(x_device),device=self.device_,generator=generator)
                for indices in order.split(self.batch_size):
                    yield x_device[indices],y_device[indices]
        else:
            loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(target)),
                batch_size=self.batch_size,shuffle=True,
                generator=torch.Generator().manual_seed(self.seed))
            batches=lambda:iter(loader)
        scaler=torch.amp.GradScaler("cuda",enabled=gpu and self.mixed_precision)
        for _ in range(self.epochs):
            self.model_.train()
            for xb,yb in batches():
                xb=xb.to(self.device_,non_blocking=gpu); yb=yb.to(self.device_,non_blocking=gpu)
                with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=gpu and self.mixed_precision):
                    logits=self.model_(xb).squeeze(-1)
                    truth=yb[:,None].expand_as(logits)
                    loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,truth) if classification
                          else torch.nn.functional.mse_loss(logits,truth))
                if not torch.isfinite(loss): raise FloatingPointError("TabM损失非有限")
                optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward()
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(self.model_.parameters(),1.)
                scaler.step(optimizer); scaler.update()
        return self

    def _predict_values(self,X):
        import torch
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self,"model_")
        x=np.asarray(X,dtype=np.float32); results=[]; self.model_.eval()
        with torch.no_grad():
            for start in range(0,len(x),self.batch_size):
                xb=torch.from_numpy(x[start:start+self.batch_size]).to(self.device_,non_blocking=self.device_=="cuda")
                with torch.autocast(device_type="cuda",dtype=torch.float16,
                                    enabled=self.device_=="cuda" and self.mixed_precision):
                    z=self.model_(xb).squeeze(-1)
                    if isinstance(self,ClassifierMixin): z=torch.sigmoid(z)
                results.append(z.mean(dim=1).cpu().numpy())
        return np.concatenate(results)*self.scale_+self.mean_


class TabMRegressor(RegressorMixin,_TabMBase):
    def predict(self,X): return self._predict_values(X)


class TabMClassifier(ClassifierMixin,_TabMBase):
    def predict_proba(self,X):
        p=self._predict_values(X)
        return np.column_stack([1-p,p])
    def predict(self,X): return (self._predict_values(X)>=.5).astype(int)


Task = Literal["regression", "classification"]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    task: Task
    estimator: Any
    search_space: dict[str, list[Any]]
    family: str
    supplementary: bool = False


def _optional_lightgbm(task: Task, seed: int) -> Any:
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("LightGBM未安装；请安装requirements.txt或从模型列表移除") from exc
    common = dict(
        n_estimators=500, learning_rate=0.03, num_leaves=31, max_depth=-1,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )
    return LGBMRegressor(**common) if task == "regression" else LGBMClassifier(**common)


def _optional_xgboost(task: Task, seed: int) -> Any:
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("XGBoost未安装；请安装requirements.txt或从模型列表移除") from exc
    common = dict(
        n_estimators=500, learning_rate=0.03, max_depth=6, min_child_weight=3,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=seed, n_jobs=1, tree_method="hist", device="cuda",
    )
    if task == "regression":
        return XGBRegressor(objective="reg:squarederror", **common)
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)


def _optional_catboost(task: Task, seed: int) -> Any:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CatBoost未安装；请安装requirements.txt或从模型列表移除") from exc
    backend=os.environ.get("MSNET_CATBOOST_BACKEND", "GPU").upper()
    if backend not in {"GPU", "CPU"}:
        raise ValueError("MSNET_CATBOOST_BACKEND只能为GPU或CPU")
    common = dict(
        iterations=500, depth=7, learning_rate=0.03, l2_leaf_reg=5.0,
        random_seed=seed, verbose=False, allow_writing_files=False,
    )
    if backend == "GPU":
        # On Windows/WDDM, desktop processes reserve part of a laptop GPU.
        # This cap changes only allocator headroom, not the model/search space.
        common.update(task_type="GPU", devices="0", gpu_ram_part=.25)
    else:
        # Explicit recovery mode for a GPU out-of-memory failure.  CatBoost's
        # objective and hyperparameter search remain unchanged.
        common.update(task_type="CPU", thread_count=-1)
    return CatBoostRegressor(loss_function="RMSE", **common) if task == "regression" else CatBoostClassifier(
        loss_function="Logloss", **common
    )


def available_model_ids(include_supplementary: bool = False) -> list[str]:
    main = ["linear", "random_forest", "lightgbm", "xgboost", "catboost"]
    return [*main, "mlp", "tabm"] if include_supplementary else main


def build_spec(
    model_id: str,
    task: Task,
    preprocessor: Any,
    seed: int,
) -> ModelSpec:
    """创建模型与有限、预先定义的搜索空间。"""
    if task not in {"regression", "classification"}:
        raise ValueError("task必须为regression或classification")

    prefix = "model__"
    supplementary = False
    if model_id == "linear":
        if task == "regression":
            estimator = Ridge(max_iter=5000)
            space = {f"{prefix}alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
            family = "Ridge"
        else:
            estimator = LogisticRegression(max_iter=5000, solver="liblinear", random_state=seed)
            space = {f"{prefix}C": [0.01, 0.1, 1.0, 10.0, 100.0]}
            family = "LogisticRegression"
    elif model_id == "random_forest":
        common = dict(n_estimators=500, random_state=seed, n_jobs=-1)
        estimator = RandomForestRegressor(**common) if task == "regression" else RandomForestClassifier(
            class_weight=None, **common
        )
        space = {
            f"{prefix}max_depth": [None, 8, 16],
            f"{prefix}min_samples_leaf": [1, 3, 8],
            f"{prefix}max_features": ["sqrt", 0.7, 1.0],
        }
        family = "RandomForest"
    elif model_id == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(random_state=seed,early_stopping=False) if task == "regression" else HistGradientBoostingClassifier(
            random_state=seed,early_stopping=False
        )
        space = {
            f"{prefix}learning_rate": [0.03, 0.06, 0.10],
            f"{prefix}max_leaf_nodes": [15, 31, 63],
            f"{prefix}l2_regularization": [0.0, 1.0, 5.0],
        }
        family = "HistGradientBoosting"
    elif model_id == "lightgbm":
        estimator = _optional_lightgbm(task, seed)
        space = {
            f"{prefix}num_leaves": [15, 31, 63],
            f"{prefix}min_child_samples": [10, 20, 50],
            f"{prefix}reg_lambda": [0.0, 1.0, 5.0],
        }
        family = "LightGBM"
    elif model_id == "xgboost":
        estimator = _optional_xgboost(task, seed)
        space = {
            f"{prefix}max_depth": [3, 6, 9],
            f"{prefix}min_child_weight": [1, 3, 8],
            f"{prefix}reg_lambda": [0.1, 1.0, 5.0],
        }
        family = "XGBoost"
    elif model_id == "catboost":
        estimator = _optional_catboost(task, seed)
        space = {
            f"{prefix}depth": [5, 7, 9],
            f"{prefix}l2_leaf_reg": [1.0, 5.0, 10.0],
        }
        family = "CatBoost"
    elif model_id == "mlp":
        supplementary = True
        common = dict(
            hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4,
            learning_rate_init=1e-3, max_iter=400, batch_size=1024, early_stopping=False,
            validation_fraction=0.15, n_iter_no_change=30, random_state=seed,
        )
        estimator = MLPRegressor(**common) if task == "regression" else MLPClassifier(**common)
        space = {
            f"{prefix}hidden_layer_sizes": [(128, 64), (256, 128), (128, 128, 64)],
            f"{prefix}alpha": [1e-5, 1e-4, 1e-3],
            f"{prefix}learning_rate_init": [3e-4, 1e-3],
        }
        family = "MLP"
    elif model_id == "tabm":
        import importlib.util
        if importlib.util.find_spec("tabm") is None: raise RuntimeError("TabM未安装；不允许以MLP冒名替代")
        estimator=(TabMRegressor if task=="regression" else TabMClassifier)(seed=seed)
        space={f"{prefix}width":[128,256],f"{prefix}depth":[2,3],
               f"{prefix}learning_rate":[.0005,.002],f"{prefix}epochs":[60,120]}
        family="TabM"; supplementary=True
    elif model_id == "elastic_net" and task == "regression":
        estimator = ElasticNet(max_iter=10000, random_state=seed)
        space = {
            f"{prefix}alpha": [0.001, 0.01, 0.1, 1.0],
            f"{prefix}l1_ratio": [0.1, 0.5, 0.9],
        }
        family = "ElasticNet"
    else:
        raise ValueError(f"未知或不适用于{task}的模型: {model_id}")

    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    return ModelSpec(
        model_id=model_id,
        task=task,
        estimator=pipeline,
        search_space=space,
        family=family,
        supplementary=supplementary,
    )


def positive_class_probability(fitted_pipeline: Any, frame: Any) -> Any:
    probabilities = fitted_pipeline.predict_proba(frame)
    classes = list(fitted_pipeline.named_steps["model"].classes_)
    if 1 not in classes:
        raise RuntimeError("分类器训练结果中没有正类1")
    return probabilities[:, classes.index(1)]

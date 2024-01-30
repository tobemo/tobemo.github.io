from pathlib import Path
from warnings import warn

import numpy as np
import onnxruntime
import yaml


class OnnxModel:
    model: onnxruntime.InferenceSession
    mean: float = 0
    std: float = 1
    def __init__(self, fp: str) -> None:
        self.fp_model = Path(fp) / "model_opt.onnx"
        assert self.fp_model.exists()
        self.model = onnxruntime.InferenceSession(self.fp_model)
        self.input_name = self.model.get_inputs()[0].name
        self.input_length = self.model.get_inputs()[0].shape[-1]
        
        self.fp_data = Path(fp) / "data.yaml"
        if not self.fp_data.exists():
            warn("YAML file with pre-processing values not found. Trying without.")
            return
        with open(self.fp_data, "r") as f:
            used_config = yaml.safe_load(f)
            self.mean = used_config['mean']
            self.std = used_config['std']

    def __repr__(self) -> str:
        return str(self.fp_model)
    
    def _format_X(self, X: np.ndarray) -> np.ndarray:
        # type setting
        X = X.astype(np.float32)
        # normalizing
        X = (X - self.mean) / self.std
        # shaping
        if X.ndim == 2:
            X = np.expand_dims(X, 0)
        if X.shape[-1] > self.input_length:
            X = X[:,:,:self.input_length]
        if X.shape[-1] < self.input_length:
            X = np.pad(X, [(0,0), (0, 0), (0, self.input_length - X.shape[-1])])
        
        return X
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        X = self._format_X(X)        
        logits = self.model.run(None, {self.input_name: X})[0]
        return logits[0]
    
    def _proba(self, logits: np.ndarray) -> np.ndarray:
        return np.exp(logits)
    
    def proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba(self.forward(X))
    
    def _predict(self, y: np.ndarray) -> int:
        return np.argmax(y)
    
    def predict(self, X: np.ndarray) -> int:
        return self._predict(self.proba(X))


class ModelList:
    models: list[OnnxModel]
    def __init__(self, fps: str | list) -> None:
        self.models = [OnnxModel(fp) for fp in fps]
    
    def proba(self, X: np.ndarray) -> np.ndarray:
        ps = np.stack([model.proba(X) for model in self.models])
        p = ps.mean(axis=0)
        return p

    def _predict(self, y: np.ndarray) -> int:
        return np.argmax(y)
    
    def predict(self, X: np.ndarray) -> int:
        return self._predict(self.proba(X))

    def __getitem__(self, key):
        return self.models.__getitem__(key)
    
    def __repr__(self) -> str:
        return f"[{', '.join([str(m) for m in self.models])}]"


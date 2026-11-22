"""Optional covariance-shift diagnostic for selected subsets.

Measures ||Cov(selected) - Cov(real train)||_F / ||Cov(real train)||_F in the k-dim PCA
space of the real embeddings (same convention as covariance_matching.py). Needs the
git-ignored embedding files; returns None when they are missing so the pipeline still runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


class CovarianceDiagnostic:
    def __init__(self, synth_emb_path=config.SYNTH_EMBEDDINGS_PT, real_emb_path=config.REAL_TRAIN_EMBEDDINGS_PT,
                 pca_dim: int = 64):
        import torch

        self.synth = torch.load(synth_emb_path, map_location="cpu", weights_only=True).numpy()
        real = torch.load(real_emb_path, map_location="cpu", weights_only=True).numpy().astype(np.float64)
        self.mu = real.mean(axis=0)
        _, _, vt = np.linalg.svd(real - self.mu, full_matrices=False)
        self.basis = vt[:pca_dim].T
        z = (real - self.mu) @ self.basis
        self.sigma_t = np.cov(z, rowvar=False, bias=True)
        self.sigma_norm = float(np.linalg.norm(self.sigma_t))

    @classmethod
    def try_create(cls, pca_dim: int = 64):
        if not (Path(config.SYNTH_EMBEDDINGS_PT).is_file() and Path(config.REAL_TRAIN_EMBEDDINGS_PT).is_file()):
            return None
        try:
            return cls(pca_dim=pca_dim)
        except Exception as exc:  # torch missing, corrupt file, ...
            print(f"[diagnostics] covariance diagnostic disabled: {exc}")
            return None

    def relative_distance(self, source_row_idx) -> float:
        idx = np.asarray(source_row_idx, dtype=np.int64)
        if len(idx) < 2:
            return float("nan")
        z = (self.synth[idx].astype(np.float64) - self.mu) @ self.basis
        cov = np.cov(z, rowvar=False, bias=True)
        return float(np.linalg.norm(cov - self.sigma_t) / self.sigma_norm)

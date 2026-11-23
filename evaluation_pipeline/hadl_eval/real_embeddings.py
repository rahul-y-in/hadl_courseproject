"""Compute the real-train embeddings (886 x 768) needed by the covariance diagnostic.

The file embedding_generation/odia_real_train_embeddings.pt is git-ignored and is not on the
team Drive, so the diagnostic recreates it when it is missing. The embedding recipe is taken
directly from Vanshika's covariance_matching/covariance_matching.py (model name, max length,
input text format, mean pooling + L2 norm), so it cannot drift from what covariance matching
used. Before embedding the real rows, a parity check re-embeds the first synthetic rows and
compares them with the stored synthetic embeddings (cosine >= 0.99), exactly like her script.

Needs: transformers, torch, internet access to download intfloat/multilingual-e5-base (~1.1 GB)
the first time. CPU is fine (~4 minutes); a GPU is used automatically if present.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

CM_SCRIPT = config.PROJECT_ROOT / "covariance_matching" / "covariance_matching.py"


def _load_cm_module():
    """Import covariance_matching.py by path without writing __pycache__ into that folder."""
    if not CM_SCRIPT.is_file():
        raise FileNotFoundError(f"{CM_SCRIPT} not found (needed for the embedding recipe)")
    spec = importlib.util.spec_from_file_location("hadl_covariance_matching", CM_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def compute_real_train_embeddings(synth_emb: np.ndarray, out_path: str | Path = config.REAL_TRAIN_EMBEDDINGS_PT,
                                  n_parity: int = 8, min_cosine: float = 0.99) -> np.ndarray:
    """Embed the real IndicQA train rows with the covariance-matching recipe and save them.

    synth_emb: the full synthetic embedding matrix (odia_embeddings.pt), rows aligned with
               or_synthetic_filtered.parquet - used for the parity check.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    cm = _load_cm_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32 if device.type == "cuda" else 8
    print(f"[real embeddings] loading {cm.MODEL_NAME} on {device} (first run downloads ~1.1 GB)")
    tokenizer = AutoTokenizer.from_pretrained(cm.MODEL_NAME)
    model = AutoModel.from_pretrained(cm.MODEL_NAME).to(device).eval()

    # Parity check: re-embed the first synthetic rows and compare with the stored vectors.
    probe = pd.read_parquet(config.SYNTH_FILTERED, columns=["question", "paragraph"]).head(n_parity)
    probe_texts = [cm.build_text(q, p) for q, p in zip(probe["question"], probe["paragraph"])]
    probe_emb = cm.embed_texts(probe_texts, tokenizer, model, device, batch_size, "Recipe parity check").numpy()
    stored = np.asarray(synth_emb[:n_parity], dtype=np.float32)
    cosine = (probe_emb * stored).sum(axis=1) / (np.linalg.norm(probe_emb, axis=1) * np.linalg.norm(stored, axis=1))
    print("[real embeddings] parity cosine (re-embedded vs stored):", np.round(cosine, 5))
    if cosine.min() < min_cosine:
        raise RuntimeError("Embedding recipe does not reproduce odia_embeddings.pt (cosine < "
                           f"{min_cosine}); not computing real embeddings that would not be comparable.")

    real = pd.read_json(config.REAL_DIR / "train.jsonl", lines=True)
    texts = [cm.build_text(q, p) for q, p in zip(real["question"], real["passage"])]
    real_emb = cm.embed_texts(texts, tokenizer, model, device, batch_size, "Embedding real train rows")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(real_emb, out_path)
    print(f"[real embeddings] saved {tuple(real_emb.shape)} to {out_path}")
    return real_emb.numpy()

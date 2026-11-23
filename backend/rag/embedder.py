"""
Query embedder for BAAI/bge-base-en-v1.5 using ONNX Runtime + HF tokenizers.

Replaces sentence-transformers/torch (too large for Vercel) while producing the same
vectors: CLS-token pooling followed by L2 normalisation. The ONNX weights are downloaded
once per instance into MODEL_CACHE_DIR (/tmp on Vercel) and reused by warm invocations.
"""

import os
import threading
import urllib.request
from pathlib import Path

import numpy as np

from . import config

_lock = threading.Lock()
_session = None
_tokenizer = None


def _download(remote_path: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://huggingface.co/{config.EMBED_MODEL_REPO}/resolve/"
        f"{config.EMBED_MODEL_REVISION}/{remote_path}"
    )
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def _load():
    global _session, _tokenizer
    if _session is not None:
        return _session, _tokenizer
    with _lock:
        if _session is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            cache = config.MODEL_CACHE_DIR
            model_path = cache / Path(config.EMBED_MODEL_FILE).name
            tokenizer_path = cache / "tokenizer.json"
            _download("tokenizer.json", tokenizer_path)
            _download(config.EMBED_MODEL_FILE, model_path)

            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer.enable_truncation(max_length=config.EMBED_MAX_TOKENS)
            tokenizer.no_padding()

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
            _tokenizer = tokenizer
    return _session, _tokenizer


def warm_up() -> None:
    """Download (if needed) and load the model."""
    _load()


def is_loaded() -> bool:
    return _session is not None


def embed_query(text: str) -> np.ndarray:
    """Return a normalised float32 vector (768,) for one query string."""
    session, tokenizer = _load()
    encoding = tokenizer.encode(text)
    input_ids = np.asarray([encoding.ids], dtype=np.int64)
    feed = {
        "input_ids": input_ids,
        "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
    }
    if any(i.name == "token_type_ids" for i in session.get_inputs()):
        feed["token_type_ids"] = np.asarray([encoding.type_ids], dtype=np.int64)

    last_hidden_state = session.run(None, feed)[0]
    vector = last_hidden_state[0, 0].astype(np.float32)  # CLS pooling (bge)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)

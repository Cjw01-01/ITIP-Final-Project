"""
DistilBERT Track Classifier — Inference Module (proposal §5.5).

Predicts BMW internship track for a candidate text snippet.
Returns top-K predictions with confidence scores.

The model is loaded lazily on first call and cached in memory.
If the model files are not found, gracefully returns None (Agent A
falls back to GPT-based classification).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent.parent.parent
# Docker Compose mounts ./models → /itip/models (legacy images used /app/models)
for _cand in (Path("/itip/models/track_classifier"), Path("/app/models/track_classifier")):
    if _cand.exists() and (_cand / "config.json").exists():
        MODEL_DIR = _cand
        break
else:
    MODEL_DIR = _project_root / "models" / "track_classifier"
_model_cache: dict[str, Any] = {}


def _load_model() -> bool:
    """Load model + tokenizer into _model_cache. Returns True on success."""
    if "model" in _model_cache:
        return _model_cache["model"] is not None

    try:
        import torch
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

        if not (MODEL_DIR / "config.json").exists():
            logger.warning("DistilBERT model not found at %s — skipping classifier", MODEL_DIR)
            _model_cache["model"] = None
            return False

        tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
        model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        label_map_path = MODEL_DIR / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, encoding="utf-8") as f:
                label_map = json.load(f)
            id2label = {int(k): v for k, v in label_map["id2label"].items()}
        else:
            id2label = model.config.id2label

        _model_cache.update({
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "id2label": id2label,
        })
        logger.info("DistilBERT track classifier loaded from %s", MODEL_DIR)
        return True

    except ImportError:
        logger.warning("transformers/torch not installed — DistilBERT classifier unavailable")
        _model_cache["model"] = None
        return False
    except Exception as e:
        logger.error("Failed to load DistilBERT model: %s", e)
        _model_cache["model"] = None
        return False


def predict_track(text: str, top_k: int = 3) -> list[dict] | None:
    """
    Predict BMW track for a candidate text.

    Returns list of dicts: [{"track": "AI", "confidence": 0.95}, ...]
    sorted by confidence descending.

    Returns None if the model is unavailable.
    """
    if not _load_model():
        return None

    import torch

    model = _model_cache["model"]
    tokenizer = _model_cache["tokenizer"]
    device = _model_cache["device"]
    id2label = _model_cache["id2label"]

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()

    results = []
    for idx in torch.argsort(probs, descending=True)[:top_k]:
        results.append({
            "track": id2label[idx.item()],
            "confidence": round(probs[idx].item(), 4),
        })

    return results


def predict_track_single(text: str) -> tuple[str, float] | None:
    """Convenience: return (track_name, confidence) for top-1 prediction."""
    preds = predict_track(text, top_k=1)
    if not preds:
        return None
    return preds[0]["track"], preds[0]["confidence"]

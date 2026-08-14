import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

ROOT = Path(__file__).parent
IMAGES_DIR = ROOT / "images"

_INDEX_LOADED = False
_EMBEDDINGS = None
_METAS = None
_MODEL = None
_PROCESSOR = None


def _ensure_index(index_dir: str = None):
    global _INDEX_LOADED, _EMBEDDINGS, _METAS
    if _INDEX_LOADED:
        return
    base = Path(index_dir or ROOT / "index")
    emb_path = base / "embeddings.npy"
    metas_path = base / "metas.json"
    if not emb_path.exists() or not metas_path.exists():
        raise FileNotFoundError("Index files not found. Build the index first.")
    _EMBEDDINGS = np.load(emb_path)
    with open(metas_path, "r", encoding="utf-8") as f:
        _METAS = json.load(f)
    _INDEX_LOADED = True


def _ensure_model(model_name: str = "openai/clip-vit-base-patch32"):
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return
    _PROCESSOR = AutoProcessor.from_pretrained(model_name)
    _MODEL = AutoModel.from_pretrained(model_name)


def _resolve_path(p: str) -> str:
    """Resolve a stored meta path to a real file on this machine.

    metas.json may contain:
      - a path that's already relative to images/ (preferred, portable)
      - a stale absolute path baked in on a different machine at build time

    We try, in order: the path as-is, the path joined onto IMAGES_DIR, and
    finally images/<basename(p)> as a last-resort fallback.
    """
    if not p:
        return p

    # 1. Path works as given (either already correct, or a valid absolute path)
    if os.path.exists(p):
        return p

    # 2. Treat it as relative to the images/ directory
    candidate = IMAGES_DIR / p
    if candidate.exists():
        return str(candidate)

    # 3. Fall back to matching just the filename inside images/
    candidate = IMAGES_DIR / os.path.basename(p)
    if candidate.exists():
        return str(candidate)

    # Give up — caller will treat this as a missing file
    return p


def _image_embedding_from_path(p: str):
    _ensure_model()
    image = Image.open(p).convert("RGB")
    inputs = _PROCESSOR(images=image, return_tensors="pt")
    out = _MODEL.get_image_features(**inputs)
    if hasattr(out, "cpu"):
        emb = out.cpu().detach().numpy().reshape(-1)
    elif hasattr(out, "image_embeds"):
        emb = out.image_embeds.detach().cpu().numpy().reshape(-1)
    elif hasattr(out, "pooler_output"):
        emb = out.pooler_output.detach().cpu().numpy().reshape(-1)
    else:
        try:
            emb = np.asarray(out).reshape(-1)
        except Exception:
            raise RuntimeError("Unable to extract image embedding from model output")
    return emb


def _color_histogram(path: str, bins: int = 32) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    hist = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 255))
        hist.append(h)
    hist = np.concatenate(hist).astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


def _combine_scores(emb_dists: np.ndarray, hist_dists: np.ndarray, alpha: float = 0.7):
    def norm(x):
        x = np.array(x, dtype=np.float32)
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-9:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    emb_n = norm(emb_dists)
    hist_n = norm(hist_dists)
    combined = alpha * emb_n + (1 - alpha) * hist_n
    return combined


def make_collage(paths: List[str], thumb_size: int = 256, cols: int = 3) -> Image.Image:
    imgs = [Image.open(_resolve_path(p)).convert("RGB") for p in paths]
    rows = (len(imgs) + cols - 1) // cols
    w = cols * thumb_size
    h = rows * thumb_size
    collage = Image.new("RGB", (w, h), (255, 255, 255))
    for i, im in enumerate(imgs):
        im = im.copy()
        im.thumbnail((thumb_size, thumb_size))
        x = (i % cols) * thumb_size
        y = (i // cols) * thumb_size
        collage.paste(im, (x, y))
    return collage


def search_image(image_path: str, top_k: int = 5, re_rank: bool = True) -> List[Dict]:
    """Return a list of results with improved scoring and metadata.

    Each result: {path, name, sku, emb_dist, hist_dist, combined, similarity}

    Self-match removal: If the query image exists in the index, it will be filtered out.
    """
    _ensure_index()
    q_emb = _image_embedding_from_path(image_path)
    dists = np.linalg.norm(_EMBEDDINGS - q_emb.reshape(1, -1), axis=1)

    self_match_indices = set(np.where(dists < 0.01)[0])
    all_idxs = np.argsort(dists)
    idxs = [i for i in all_idxs if i not in self_match_indices][: max(top_k * 3, top_k)]

    candidates = []
    hist_q = _color_histogram(image_path)
    hist_dists = []
    emb_dists = []

    for i in idxs:
        meta = _METAS[i]
        raw_p = meta.get("path") or meta.get("url")
        p = _resolve_path(raw_p)
        if not p or not os.path.exists(p):
            hist_d = 1.0
        else:
            hist = _color_histogram(p)
            hist_d = float(np.linalg.norm(hist_q - hist))
        hist_dists.append(hist_d)
        emb_dists.append(float(dists[i]))
        candidates.append((i, meta))

    if re_rank:
        combined = _combine_scores(np.array(emb_dists), np.array(hist_dists), alpha=0.75)
    else:
        combined = np.array(emb_dists)

    results = []
    for score, (i, meta), emb_d, hist_d in zip(combined, candidates, emb_dists, hist_dists):
        sim = 1.0 - float(score)
        sim = max(0.0, min(1.0, sim))
        resolved_path = _resolve_path(meta.get("path") or meta.get("url"))
        results.append(
            {
                "index": int(i),
                "path": resolved_path,
                "name": meta.get("name") or meta.get("title") or "",
                "sku": meta.get("sku") or meta.get("id") or "",
                "emb_dist": emb_d,
                "hist_dist": hist_d,
                "combined": float(score),
                "similarity": float(sim),
            }
        )

    results = sorted(results, key=lambda r: r["combined"])[:top_k]
    return results

import io
import json
import os
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

ROOT = Path(__file__).parent
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

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


def _is_url(p: str) -> bool:
    return isinstance(p, str) and p.startswith(("http://", "https://"))


def _extract_basename(p: str) -> str:
    """Basename extraction that works for BOTH Windows ('C:\\a\\b.jpg') and
    Unix ('/a/b.jpg') style paths, regardless of which OS we're running on.
    os.path.basename() alone isn't enough: on Linux it doesn't know '\\' is
    a separator, so a Windows path baked into metas.json on another machine
    would otherwise come back unsplit.
    """
    if not p:
        return p
    return re.split(r"[\\/]", p)[-1]


def _download_and_cache(url: str) -> str:
    """Download a remote image URL into images/ and return the local path.
    If download fails, return the original URL unchanged so callers can
    still fall back to rendering it directly (e.g. Streamlit can display
    an image straight from a URL)."""
    fname = _extract_basename(url.split("?")[0]) or "download"
    stem = os.path.splitext(fname)[0] or "download"
    target = IMAGES_DIR / f"{stem}.jpg"
    if target.exists():
        return str(target)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(target, format="JPEG")
        return str(target)
    except Exception:
        return url


def _resolve_path(p: str) -> str:
    """Resolve a stored meta path to something we can actually open.

    Handles three cases seen in the wild in this project's metas.json:
      1. A remote URL (no local file was ever downloaded for this row)
         -> download + cache it locally, or return the URL as a fallback.
      2. A stale absolute path baked in on a different machine/OS
         (e.g. a Windows path used from Linux/Mac)
         -> fall back to images/<basename>, extracted OS-agnostically.
      3. A path relative to images/, or one that already exists as-is.
    """
    if not p:
        return p

    if _is_url(p):
        return _download_and_cache(p)

    if os.path.exists(p):
        return p

    candidate = IMAGES_DIR / p
    if candidate.exists():
        return str(candidate)

    candidate = IMAGES_DIR / _extract_basename(p)
    if candidate.exists():
        return str(candidate)

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
    imgs = []
    for p in paths:
        resolved = _resolve_path(p)
        try:
            if _is_url(resolved):
                resp = requests.get(resolved, timeout=10)
                resp.raise_for_status()
                imgs.append(Image.open(io.BytesIO(resp.content)).convert("RGB"))
            else:
                imgs.append(Image.open(resolved).convert("RGB"))
        except Exception:
            continue

    if not imgs:
        raise RuntimeError("No valid images available to build a collage")

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
    resolved_paths = []

    for i in idxs:
        meta = _METAS[i]
        raw_p = meta.get("path") or meta.get("url")
        p = _resolve_path(raw_p)
        resolved_paths.append(p)

        if not p or _is_url(p) or not os.path.exists(p):
            # either resolution failed, or it's still a bare URL
            # (download failed) — can't compute a real histogram for it
            hist_d = 1.0
        else:
            try:
                hist = _color_histogram(p)
                hist_d = float(np.linalg.norm(hist_q - hist))
            except Exception:
                hist_d = 1.0

        hist_dists.append(hist_d)
        emb_dists.append(float(dists[i]))
        candidates.append((i, meta))

    if re_rank:
        combined = _combine_scores(np.array(emb_dists), np.array(hist_dists), alpha=0.75)
    else:
        combined = np.array(emb_dists)

    results = []
    for score, (i, meta), emb_d, hist_d, resolved_p in zip(
        combined, candidates, emb_dists, hist_dists, resolved_paths
    ):
        sim = 1.0 - float(score)
        sim = max(0.0, min(1.0, sim))
        results.append(
            {
                "index": int(i),
                "path": resolved_p,
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

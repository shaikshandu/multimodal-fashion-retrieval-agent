from typing import List, Dict
import os
import random
from PIL import Image
try:
    from . import tool
except Exception:
    import tool


def _format_price(meta: Dict) -> str:
    for k in ("price", "mrp", "price_text", "amount", "sale_price"):
        v = meta.get(k)
        if v:
            return str(v)
    return ""


def _generate_natural_explanation(results: List[Dict]) -> str:
    """Generate a natural, conversational explanation based on search results.
    
    Args:
        results: List of result dicts with similarity scores
    
    Returns:
        A natural language explanation of the search results
    """
    if not results:
        return "I couldn't find any similar sarees. Please try another image."
    
    n = len(results)
    top_similarity = results[0].get('similarity', 0) * 100 if results else 0

    if top_similarity >= 85:
        intro = f"Excellent! I found {n} highly similar saree{'s' if n != 1 else ''} that match your style perfectly."
    elif top_similarity >= 70:
        intro = f"Great! I discovered {n} visually similar saree{'s' if n != 1 else ''} based on color, pattern, and design."
    elif top_similarity >= 50:
        intro = f"I found {n} saree{'s' if n != 1 else ''} with similar aesthetic elements like color, weave, and design patterns."
    else:
        intro = f"I located {n} saree{'s' if n != 1 else ''} that share some visual characteristics with your query image."

    details = ""
    if results:
        first_result = results[0]
        name = first_result.get('name', 'Saree')
        if name and name.strip():
            details = f" The closest match is {name}."
    
    return intro + details


def search_similar_sarees(image_path: str, top_k: int = 5) -> Dict:
    """Agent wrapper: search and produce an explanatory assistant-style response.

    Returns dict with keys: explanation (str), results (list of dicts), collage_path (optional)
    Each result dict: {index, path, name, sku, similarity (0..1), price (or '')}
    """
    results = tool.search_image(image_path, top_k=top_k, re_rank=True)
    formatted = []
    for r in results:
        meta = {}
        meta['index'] = r.get('index')
        meta['path'] = r.get('path')
        meta['name'] = r.get('name') or ''
        meta['sku'] = r.get('sku') or ''
        meta['similarity'] = float(r.get('similarity', 0.0))
        
        price = ''
        try:
            price = ''
        except Exception:
            price = ''
        meta['price'] = price
        formatted.append(meta)

    explanation = _generate_natural_explanation(formatted)
    collage_path = None
    try:
        paths = [r['path'] for r in formatted if r.get('path')]
        if paths:
            collage = tool.make_collage(paths, thumb_size=256, cols=min(3, len(paths)))
            outp = os.path.join(os.path.dirname(image_path), 'results_collage.jpg')
            collage.save(outp)
            collage_path = outp
    except Exception:
        collage_path = None

    return {"explanation": explanation, "results": formatted, "collage_path": collage_path}

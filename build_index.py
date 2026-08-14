import os
import json
import math
from PIL import Image, ImageDraw
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

ROOT = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(ROOT, 'images')
INDEX_DIR = os.path.join(ROOT, 'index')

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


def make_synthetic_saree(name, idx, size=(224, 224)):
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    base = ((idx * 37) % 255, (idx * 91) % 255, (idx * 151) % 255)
    draw.rectangle([0, 0, size[0], size[1]], fill=base)
    for i in range(0, size[0], 20):
        shade = ((base[0] + i) % 255, (base[1] + i * 2) % 255, (base[2] + i * 3) % 255)
        draw.line([(i, 0), (i, size[1])], fill=shade, width=4)
    for r in range(5):
        cx = int(size[0] * (0.2 + 0.6 * (r / 4)))
        cy = size[1] // 2
        radius = 10 + r * 6
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(255, 255, 255), width=3)
    fname = f'{name}_{idx}.jpg'
    path = os.path.join(IMAGES_DIR, fname)
    img.save(path)
    return fname  # return just the filename, not the absolute path


def build(sample_count=80):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    filenames = []
    metas = []
    for i in range(sample_count):
        name = 'saree'
        fname = make_synthetic_saree(name, i)
        filenames.append(fname)
        # Store a path relative to images/ so the index is portable across
        # machines and repo clones. tool.py resolves this against IMAGES_DIR.
        metas.append({'id': i, 'path': fname, 'name': f'{name}_{i}'})

    embeddings = []
    for fname in filenames:
        p = os.path.join(IMAGES_DIR, fname)
        img = Image.open(p).convert('RGB')
        inputs = processor(images=img, return_tensors='pt').to(device)
        with torch.no_grad():
            emb_out = model.get_image_features(**inputs)
        if hasattr(emb_out, 'cpu'):
            vec = emb_out.cpu().numpy().reshape(-1)
        else:
            if hasattr(emb_out, 'image_embeds'):
                vec = emb_out.image_embeds.cpu().numpy().reshape(-1)
            elif hasattr(emb_out, 'pooler_output'):
                vec = emb_out.pooler_output.cpu().numpy().reshape(-1)
            else:
                try:
                    t = torch.tensor(emb_out)
                    vec = t.cpu().numpy().reshape(-1)
                except Exception:
                    raise RuntimeError('Could not extract embedding tensor from model output')
        embeddings.append(vec)

    embeddings = np.stack(embeddings).astype('float32')
    np.save(os.path.join(INDEX_DIR, 'embeddings.npy'), embeddings)
    with open(os.path.join(INDEX_DIR, 'metas.json'), 'w', encoding='utf-8') as f:
        json.dump(metas, f, indent=2)

    print(f'Built index (embeddings saved) with {len(metas)} items at {INDEX_DIR}')


if __name__ == '__main__':
    build(sample_count=80)

import os
import csv
import io
import time
import argparse
from PIL import Image
import requests
from tqdm import tqdm
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

ROOT = os.path.dirname(__file__)
CSV_PATH = os.path.join(os.path.dirname(ROOT), 'byrappa_tejas_31july.csv')
IMAGES_DIR = os.path.join(ROOT, 'images')
INDEX_DIR = os.path.join(ROOT, 'index')

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


def download_image(url, target_path, timeout=10, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
            img.save(target_path, format='JPEG')
            return True
        except Exception:
            time.sleep(0.5)
    return False


def build_from_csv(sample_count=300):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    metas = []
    embeddings = []
    downloaded = 0

    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(tqdm(reader)):
            if i >= sample_count:
                break
            url = row.get('image_url')
            sku = row.get('SKU') or f'row_{i}'
            if not url:
                continue

            fname = f"{sku}.jpg"
            target = os.path.join(IMAGES_DIR, fname)
            if os.path.exists(target):
                ok = True
            else:
                ok = download_image(url, target)
            if not ok:
                continue

            try:
                img = Image.open(target).convert('RGB')
                inputs = processor(images=img, return_tensors='pt').to(device)
                with torch.no_grad():
                    emb_out = model.get_image_features(**inputs)
                if hasattr(emb_out, 'cpu'):
                    vec = emb_out.cpu().numpy().reshape(-1)
                elif hasattr(emb_out, 'image_embeds'):
                    vec = emb_out.image_embeds.cpu().numpy().reshape(-1)
                elif hasattr(emb_out, 'pooler_output'):
                    vec = emb_out.pooler_output.cpu().numpy().reshape(-1)
                else:
                    vec = np.array(emb_out).reshape(-1)
            except Exception:
                continue

            # Store the path relative to images/ (just the filename here)
            # instead of an absolute path baked in at build time, so the
            # index still works after cloning the repo elsewhere.
            metas.append({'sku': sku, 'name': row.get('Name'), 'url': url, 'path': fname})
            embeddings.append(vec.astype('float32'))
            downloaded += 1

    if embeddings:
        emb_arr = np.stack(embeddings)
        np.save(os.path.join(INDEX_DIR, 'embeddings.npy'), emb_arr)
        import json
        with open(os.path.join(INDEX_DIR, 'metas.json'), 'w', encoding='utf-8') as jf:
            json.dump(metas, jf, indent=2)
        print(f'Saved {len(metas)} embeddings to {os.path.join(INDEX_DIR, "embeddings.npy")}')
    else:
        print('No embeddings were created (downloads may have failed).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=300, help='Number of rows to attempt')
    args = parser.parse_args()
    build_from_csv(sample_count=args.sample)

import os
import json
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


ROOT = os.path.dirname(__file__)
INDEX_DIR = os.path.join(ROOT, 'index')


def search_topk(query_path, k=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    img = Image.open(query_path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        emb_out = model.get_image_features(**inputs)
    if hasattr(emb_out, 'cpu'):
        q_emb = emb_out.cpu().numpy().astype('float32')
    else:
        if hasattr(emb_out, 'image_embeds'):
            q_emb = emb_out.image_embeds.cpu().numpy().astype('float32')
        elif hasattr(emb_out, 'pooler_output'):
            q_emb = emb_out.pooler_output.cpu().numpy().astype('float32')
        else:
            q_emb = np.array(emb_out).astype('float32')

    embeddings = np.load(os.path.join(INDEX_DIR, 'embeddings.npy'))
    diffs = embeddings - q_emb.reshape(1, -1)
    dists = np.sum(diffs * diffs, axis=1)
    idxs = np.argsort(dists)[:k]
    metas = json.load(open(os.path.join(INDEX_DIR, 'metas.json'), 'r', encoding='utf-8'))
    results = []
    for idx in idxs:
        m = metas[idx]
        results.append({'path': m['path'], 'score': float(dists[idx])})
    return results


if __name__ == '__main__':
    imgs = os.listdir(os.path.join(ROOT, 'images'))
    if not imgs:
        print('No images found. Run build_index.py first.')
    else:
        q = os.path.join(ROOT, 'images', imgs[len(imgs)//3])
        print('Query:', q)
        res = search_topk(q, k=5)
        print('Top results:')
        for r in res:
            print(r)

import streamlit as st
import os
import json
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

ROOT = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(ROOT, 'images')
INDEX_DIR = os.path.join(ROOT, 'index')

st.title('AI-Powered Saree Similarity Search')

emb_path = os.path.join(INDEX_DIR, 'embeddings.npy')
meta_path = os.path.join(INDEX_DIR, 'metas.json')

if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
    st.warning('Index not found. You can build a demo index here (generates synthetic images).')
    if st.button('Build index now (demo, may take a minute)'):
        with st.spinner('Building index...'):
            try:
                import build_index
                build_index.build(sample_count=80)
                st.success('Index built — reload the app to use it.')
                st.experimental_rerun()
            except Exception as e:
                st.error(f'Failed to build index: {e}')

import io
import tool
import agent


def _render_image(p, width=220):
    """Render an image in Streamlit whether it's a local file path or a
    remote URL (tool.py resolves URLs to a local cached copy when it can,
    but falls back to the bare URL if the download failed)."""
    if not p:
        st.write('Image not found')
        return
    if tool._is_url(p):
        # Let Streamlit fetch it client-side rather than trying to
        # PIL.Image.open() a URL string, which always fails.
        st.image(p, width=width)
        return
    resolved = tool._resolve_path(p)
    try:
        st.image(Image.open(resolved), width=width)
    except Exception:
        st.write(f'Image not found: {p}')


def render_results(results):
    st.write('Top matches:')
    cols = st.columns(3)
    paths = []
    for i, res in enumerate(results[:6]):
        p = res.get('path')
        paths.append(p)
        with cols[i % 3]:
            _render_image(p, width=220)
            sim = res.get('similarity', 0.0)
            emb_d = res.get('emb_dist', 0.0)
            hist_d = res.get('hist_dist', 0.0)
            st.markdown(f"**Similarity:** {sim*100:.1f}%")
            st.write(f"emb:{emb_d:.3f} hist:{hist_d:.3f}")
    return paths


uploaded = st.file_uploader('Upload a query image', type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded is not None:
    data = uploaded.read()
    qpath = os.path.join(ROOT, 'query_tmp.jpg')
    with open(qpath, 'wb') as f:
        f.write(data)
    st.image(Image.open(io.BytesIO(data)), caption='Query image', width=300)

    with st.spinner('Searching...'):
        try:
            results = tool.search_image(qpath, top_k=6, re_rank=True)
            paths = render_results(results)

            try:
                collage = tool.make_collage(paths, thumb_size=256, cols=3)
                buf = io.BytesIO()
                collage.save(buf, format='JPEG')
                st.download_button('Download collage', data=buf.getvalue(), file_name='results_collage.jpg')
            except Exception:
                pass

        except Exception as e:
            st.error(f"Search failed: {e}")

    if st.button('Agent: Explain results'):
        try:
            out = agent.search_similar_sarees(qpath, top_k=6)
            st.markdown('**Assistant:**')
            st.write(out.get('explanation'))

            cols = st.columns(3)
            for i, r in enumerate(out.get('results', [])[:6]):
                with cols[i % 3]:
                    _render_image(r.get('path'), width=200)
                    sim = r.get('similarity', 0.0)
                    price = r.get('price', '')
                    st.markdown(f"**{sim*100:.1f}%**")
                    if price:
                        st.write(price)

            if out.get('collage_path'):
                try:
                    st.image(tool._resolve_path(out.get('collage_path')), caption='Collage', width=600)
                except Exception:
                    pass

        except Exception as e:
            st.error(f'Agent failed: {e}')

    st.write('Raw results:')
    st.dataframe([
        {"path": r.get('path'), "similarity": f"{r.get('similarity')*100:.1f}%", "emb": r.get('emb_dist'), "hist": r.get('hist_dist')}
        for r in results
    ])

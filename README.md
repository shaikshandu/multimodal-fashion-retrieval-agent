# Saree Visual Search 🧵

A powerful visual search engine for finding similar sarees using AI-powered image embeddings and multi-modal similarity ranking.

## Features ✨

- **Visual Search**: Upload any saree image and find visually similar matches from your catalog
- **Multi-Modal Ranking**: Combines deep learning embeddings (CLIP) with color histogram analysis for accurate results
- **Self-Match Removal**: Automatically filters out the query image itself from results
- **Natural Chat Interface**: AI-powered assistant provides conversational explanations of search results
- **Similarity Scoring**: Each result includes detailed similarity metrics and distance scores
- **Collage Generation**: Automatically creates visual collages of results for easy comparison
- **Streamlit UI**: Beautiful, responsive web interface built with Streamlit

## Quick Start 🚀

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone or navigate to the project**:
   ```bash
   cd visual_search
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the search index** (first time only):
   ```bash
   python build_index.py --count 80  # Creates a demo index with synthetic images
   ```
   
   Or if you have a CSV with saree metadata:
   ```bash
   python build_index_from_csv.py --csv your_sarees.csv
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**: Navigate to `http://localhost:8501`

## Project Structure 📁

```
visual_search/
├── app.py                      # Main Streamlit application
├── agent.py                    # AI assistant for natural responses
├── tool.py                     # Core search and indexing functions
├── build_index.py              # Create search index from images
├── build_index_from_csv.py     # Create search index from CSV metadata
├── run_search_test.py          # Testing script
├── requirements.txt            # Python dependencies
├── index/                      # Search index directory (created after building)
│   ├── embeddings.npy          # CLIP embeddings for all images
│   └── metas.json              # Metadata for each image
└── images/                     # Sample images directory
```

## How It Works 🔧

### 1. **Index Building**
The system uses CLIP (Contrastive Learning Image Pre-training) to generate embeddings for each image:
- Downloads pre-trained CLIP model (`openai/clip-vit-base-patch32`)
- Processes each image to create a 512-dimensional embedding
- Stores embeddings and metadata in NumPy and JSON formats

### 2. **Visual Search**
When you upload a query image:
1. **Embedding Generation**: CLIP model converts your image to a 512-D embedding
2. **Distance Calculation**: Computes distances between query and all indexed embeddings
3. **Self-Match Removal**: Filters out identical images (distance < 0.01)
4. **Multi-Modal Ranking**: 
   - Combines embedding distances (75% weight) with color histogram distances (25% weight)
   - Normalizes scores for fair comparison
5. **Result Retrieval**: Returns top K results sorted by combined similarity score

### 3. **AI Explanation**
The agent generates natural language explanations based on:
- Number of results found
- Quality of top match (similarity score)
- Metadata like product name and price

## API Reference 📚

### `tool.search_image(image_path, top_k=5, re_rank=True)`
Searches for similar images in the index.

**Parameters:**
- `image_path` (str): Path to query image
- `top_k` (int): Number of results to return (default: 5)
- `re_rank` (bool): Use multi-modal re-ranking (default: True)

**Returns:** List of dicts with keys:
- `index`: Index position in embeddings array
- `path`: Path to similar image
- `name`: Product name (if available)
- `sku`: Product SKU (if available)
- `similarity`: Similarity score (0-1, higher is better)
- `emb_dist`: CLIP embedding distance
- `hist_dist`: Color histogram distance
- `combined`: Combined distance score

### `agent.search_similar_sarees(image_path, top_k=5)`
High-level wrapper that performs search and generates natural explanations.

**Returns:** Dict with keys:
- `explanation`: Natural language explanation of results
- `results`: Formatted list of similar sarees
- `collage_path`: Path to generated result collage

## Configuration 🎨

Edit `.streamlit/config.toml` to customize:
- Maximum upload file size (default: 200MB)
- Page width and layout
- Color theme

**Example config:**
```toml
[client]
maxUploadSize = 200

[theme]
primaryColor = "#FF6B9D"
backgroundColor = "#FFF0F7"
secondaryBackgroundColor = "#FFE5F0"
textColor = "#333333"
```

## Building Custom Index 🏗️

### From Directory of Images:
```python
python build_index.py --images ./my_sarees/ --output ./index
```

### From CSV File:
```python
python build_index_from_csv.py --csv sarees.csv --image-col image_path
```

**CSV should have columns:**
- `image_path`: Path to image file
- `name`: (optional) Product name
- `sku`: (optional) Product SKU
- `price`: (optional) Price information

## Performance Tips ⚡

1. **Embedding Caching**: Embeddings are cached in memory after first load
2. **Batch Processing**: For building indexes, images are processed in batches
3. **Model Optimization**: Uses `AutoModel` and `AutoProcessor` for efficient loading
4. **Result Filtering**: Only processes top candidates to reduce computation

## Troubleshooting 🔍

### "Index not found" error
- Run `python build_index.py` to create a demo index
- Or provide your own images and run the builder

### Slow search performance
- Reduce image quality/resolution for faster embedding generation
- Use smaller index (fewer images)
- Check GPU availability with `torch.cuda.is_available()`

### Memory issues
- Reduce `top_k * 3` multiplier in `tool.py` search function
- Process smaller image batches during index building

## Dependencies 📦

- `streamlit>=1.0` - Web interface
- `transformers>=4.20` - CLIP model loading
- `pillow>=8.0` - Image processing
- `numpy>=1.20` - Numerical computing
- `torch>=1.9` - Deep learning framework

## Future Enhancements 🌟

- [ ] Fine-grained ranking based on texture and pattern
- [ ] Natural language queries ("show me silk sarees with gold borders")
- [ ] Similarity graph visualization
- [ ] Export and sharing of favorites
- [ ] Mobile app support
- [ ] Real-time indexing updates

## Testing 🧪

Run the test suite:
```bash
python run_search_test.py
```

This validates:
- Index loading
- Embedding generation
- Search accuracy
- Result formatting

## License 📄

This project is provided as-is for demonstration purposes.

## Support 💬

For issues, improvements, or questions, please refer to the project documentation or create an issue in the repository.

---

**Built with ❤️ using CLIP, PyTorch, and Streamlit**

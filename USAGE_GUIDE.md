# Usage Guide - Antibiotic Resistance Pattern Recognition System

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Reyn4ldo/thesis-project02.git
cd thesis-project02

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Execute all analysis phases
python run_pipeline.py
```

This will run:
- Data cleaning and preprocessing
- Feature engineering
- Classification (6 algorithms)
- Clustering (k-Means, Hierarchical, DBSCAN)
- Association rule mining (Apriori, FP-Growth)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Statistical analysis

### 3. Launch Dashboard

```bash
# Start Streamlit dashboard
streamlit run src/deployment/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Running Individual Modules

### Data Preprocessing

```bash
# Clean data
python src/preprocessing/clean_data.py

# Engineer features
python src/preprocessing/feature_engineering.py
```

### Classification

```bash
# Train all classifiers
python src/classification/train_models.py
```

### Clustering

```bash
# Run clustering analysis
python src/clustering/cluster_analysis.py
```

### Association Rules

```bash
# Mine co-resistance patterns
python src/association_rules/mine_rules.py
```

### Dimensionality Reduction

```bash
# Generate visualizations
python src/dimensionality_reduction/visualize.py
```

### Statistical Analysis

```bash
# Perform statistical tests
python src/statistical_analysis/analyze.py
```

## Docker Deployment

### Build and Run

```bash
# Build Docker image
docker build -t antibiotic-resistance-ml .

# Run container
docker run -p 8501:8501 antibiotic-resistance-ml
```

### Using Docker Compose

```bash
# Start application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop application
docker-compose down
```

## Dashboard Features

### 1. Data Overview
- View dataset statistics
- Explore species distribution
- Analyze sample sources
- MDR prevalence overview

### 2. Classification Results
- Compare 6 ML algorithms
- View performance metrics
- Explore feature importance
- Download predictions

### 3. Clustering Analysis
- View cluster distributions
- Analyze MDR clusters
- Explore resistance patterns
- Interactive heatmaps

### 4. Association Rules
- Browse co-resistance patterns
- Filter by confidence/support/lift
- Identify ESBL patterns
- Export rules

### 5. Dimensionality Reduction
- Interactive PCA plots
- t-SNE visualizations
- UMAP embeddings
- 3D visualizations

### 6. Statistical Analysis
- Correlation heatmaps
- Hypothesis test results
- Species-resistance analysis
- MAR index correlations

### 7. Quick Prediction
- Upload CSV files
- Instant analysis
- Batch predictions
- Export results

## Output Files

All results are saved in `data/results/`:

```
data/results/
├── classification_results_species.csv
├── classification_results_mar_class.csv
├── best_model_species.pkl
├── feature_importance_species.csv
├── clustering/
│   ├── cluster_labels.csv
│   ├── kmeans_heatmap.png
│   └── dendrogram.png
├── association_rules/
│   ├── apriori_rules.csv
│   └── fpgrowth_rules.csv
├── dimensionality_reduction/
│   ├── pca_embeddings.csv
│   ├── tsne_embeddings.csv
│   └── umap_embeddings.csv
└── statistical_analysis/
    ├── antibiotic_correlation.csv
    └── correlation_heatmap.png
```

## Customization

### Adjust Classification Parameters

Edit `src/classification/train_models.py`:

```python
# Change models
self.models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,  # Increase trees
        max_depth=15,      # Increase depth
        random_state=42
    )
}
```

### Modify Clustering Settings

Edit `src/clustering/cluster_analysis.py`:

```python
# Change k-Means range
.perform_kmeans(n_clusters_range=(3, 12))

# Adjust DBSCAN
.perform_dbscan(eps=0.7, min_samples=10)
```

### Tune Association Rules

Edit `src/association_rules/mine_rules.py`:

```python
# Adjust thresholds
.mine_apriori(
    min_support=0.05,     # Lower = more rules
    min_confidence=0.7,   # Higher = stronger rules
    min_lift=1.5          # Higher = more relevant
)
```

## Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

### Issue: Out of Memory

```bash
# Reduce sample size in analysis
# Edit respective module and add:
data = data.sample(n=1000, random_state=42)
```

### Issue: Missing Data Files

```bash
# Ensure raw data exists
ls data/raw/raw_data.csv

# If missing, place your CSV in data/raw/
```

### Issue: Dashboard Not Loading

```bash
# Check if port 8501 is available
lsof -ti:8501

# Use different port
streamlit run src/deployment/app.py --server.port 8502
```

## Performance Tips

1. **Speed up training**: Reduce `n_estimators` in tree-based models
2. **Faster clustering**: Use smaller `n_clusters_range`
3. **Quick testing**: Sample data before full pipeline run
4. **Memory optimization**: Process data in batches for large datasets

## Citation

If you use this system in your research, please cite:

```
@software{antibiotic_resistance_ml,
  title={Antibiotic Resistance Pattern Recognition System},
  author={Thesis Project Team},
  year={2024},
  url={https://github.com/Reyn4ldo/thesis-project02}
}
```

## Support

For issues or questions:
- Open a GitHub issue
- Check the logs in `pipeline.log`
- Review module documentation

## License

MIT License - See LICENSE file for details

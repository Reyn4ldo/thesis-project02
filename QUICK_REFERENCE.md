# Quick Reference Card

## 🚀 Essential Commands

### Setup (One-time)
```bash
bash setup_and_test.sh
# OR
pip install -r requirements.txt
```

### Run Everything
```bash
python run_pipeline.py          # Complete analysis (all phases)
```

### Launch Dashboard
```bash
streamlit run src/deployment/app.py
# Access: http://localhost:8501
```

### Generate Report
```bash
python generate_report.py      # Creates FINAL_REPORT.md
```

### Docker
```bash
docker-compose up -d           # Start
docker-compose logs -f         # View logs
docker-compose down           # Stop
```

## 📂 Key Files

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Execute all analyses |
| `generate_report.py` | Create comprehensive report |
| `src/deployment/app.py` | Interactive dashboard |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview |
| `USAGE_GUIDE.md` | Detailed instructions |
| `PROJECT_SUMMARY.md` | Implementation details |

## 📊 Individual Modules

```bash
# Preprocessing
python src/preprocessing/clean_data.py
python src/preprocessing/feature_engineering.py

# Classification (6 algorithms)
python src/classification/train_models.py

# Clustering (k-Means, Hierarchical, DBSCAN)
python src/clustering/cluster_analysis.py

# Association Rules (Apriori, FP-Growth)
python src/association_rules/mine_rules.py

# Visualizations (PCA, t-SNE, UMAP)
python src/dimensionality_reduction/visualize.py

# Statistics (correlations, tests)
python src/statistical_analysis/analyze.py
```

## 🎯 What Each Phase Does

1. **Preprocessing**: Cleans data, engineers features, splits dataset
2. **Classification**: Trains 6 models, evaluates, selects best
3. **Clustering**: Identifies MDR clusters, creates heatmaps
4. **Association Rules**: Finds co-resistance patterns
5. **Dimensionality Reduction**: Creates 2D/3D visualizations
6. **Statistical Analysis**: Correlations, hypothesis tests

## 📈 Where to Find Results

```
data/results/
├── classification_results_*.csv       # Model performance
├── best_model_*.pkl                   # Trained models
├── clustering/
│   ├── cluster_labels.csv            # Assignments
│   └── *_heatmap.png                 # Visualizations
├── association_rules/
│   └── *_rules.csv                   # Co-resistance rules
├── dimensionality_reduction/
│   └── *.png                         # All plots
└── statistical_analysis/
    └── *_correlation.csv             # Correlations
```

## 🔍 Dashboard Features

- **Data Overview**: Statistics, distributions
- **Classification**: Model comparison, feature importance
- **Clustering**: Cluster profiles, heatmaps
- **Association Rules**: Filter by confidence/support/lift
- **Visualizations**: Interactive PCA/t-SNE/UMAP
- **Statistics**: Correlation heatmaps, test results

## ⚡ Quick Troubleshooting

**Missing modules?**
```bash
pip install -r requirements.txt
```

**Port 8501 busy?**
```bash
streamlit run src/deployment/app.py --server.port 8502
```

**Out of memory?**
- Reduce sample size in analysis files
- Use smaller n_estimators in models

**Data not found?**
```bash
ls data/raw/raw_data.csv  # Should exist
```

## 📦 Project Structure

```
thesis-project02/
├── data/
│   ├── raw/              # Input data (582 samples)
│   ├── processed/        # Cleaned data + features
│   └── results/          # All outputs
├── src/
│   ├── preprocessing/    # Data prep
│   ├── classification/   # 6 ML models
│   ├── clustering/       # 3 methods
│   ├── association_rules/# 2 algorithms
│   ├── dimensionality_reduction/  # 3 techniques
│   ├── statistical_analysis/      # Tests
│   └── deployment/       # Dashboard
├── run_pipeline.py       # Run all
├── generate_report.py    # Create report
└── setup_and_test.sh    # Setup
```

## 🎓 Algorithms Implemented

**Classification (6)**:
- Logistic Regression
- Random Forest
- XGBoost
- SVM
- k-Nearest Neighbors
- Neural Network (MLP)

**Clustering (3)**:
- k-Means
- Hierarchical
- DBSCAN

**Association Rules (2)**:
- Apriori
- FP-Growth

**Dimensionality Reduction (3)**:
- PCA
- t-SNE
- UMAP

## 💡 Tips

- Run `python run_pipeline.py` first
- Use dashboard for interactive exploration
- Check `pipeline.log` for detailed logs
- Generate report after pipeline completes
- Docker for reproducible deployment

## 📞 Support

1. Check `USAGE_GUIDE.md`
2. Review logs: `pipeline.log`
3. Read `README.md`
4. See `PROJECT_SUMMARY.md`

---

**Version**: 1.0  
**Status**: Production Ready  
**License**: MIT

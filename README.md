# Antibiotic Resistance Pattern Recognition Project

## Overview
This project implements comprehensive pattern recognition and machine learning analysis on antibiotic resistance data from bacterial isolates. The system performs classification, clustering, association rule mining, dimensionality reduction, and statistical analysis to identify resistance patterns and support antimicrobial stewardship.

## Dataset
- **Source**: Bacterial isolates from various water and fish sources
- **Species**: Enterobacter, Klebsiella, Escherichia coli
- **Sample Size**: 582 isolates
- **Features**: 22+ antibiotics with MIC values and S/I/R classifications
- **Locations**: Various regions in Central Luzon, Philippines

## Project Structure
```
thesis-project02/
├── data/
│   ├── raw/              # Original dataset
│   ├── processed/        # Cleaned and feature-engineered data
│   ├── models/           # Saved ML models
│   └── results/          # Analysis outputs
├── src/
│   ├── preprocessing/    # Data cleaning and feature engineering
│   ├── classification/   # Supervised learning models
│   ├── clustering/       # Unsupervised clustering
│   ├── association_rules/# Pattern mining
│   ├── dimensionality_reduction/  # PCA, t-SNE, UMAP
│   ├── statistical_analysis/      # Correlation and hypothesis testing
│   └── deployment/       # Streamlit dashboard
├── notebooks/            # Jupyter notebooks for exploration
├── tests/               # Unit tests
└── requirements.txt     # Python dependencies
```

## Pattern Recognition Tasks

### 1. Classification
Predict:
- Antibiotic susceptibility (S/I/R)
- Bacterial species
- MAR class (low/medium/high risk)

**Algorithms**: Logistic Regression, Random Forest, XGBoost, SVM, kNN, Neural Network

### 2. Clustering
Identify:
- MDR (Multi-Drug Resistant) clusters
- Species-level resistance patterns
- Geographic resistance hotspots

**Methods**: k-Means, Hierarchical Clustering, DBSCAN

### 3. Association Rule Mining
Discover co-resistance patterns:
- "If resistant to ampicillin → 85% resistant to cefazolin"
- "If ESBL+ → 92% resistant to ceftriaxone"

**Algorithms**: Apriori, FP-Growth

### 4. Dimensionality Reduction
Visualize hidden structures:
- PCA for linear patterns
- t-SNE for local structures
- UMAP for global topology

### 5. Statistical Pattern Recognition
- Correlation analysis (species ↔ resistance, sites ↔ resistance)
- Hypothesis testing
- Feature importance with SHAP values

## Installation

```bash
# Clone repository
git clone https://github.com/Reyn4ldo/thesis-project02.git
cd thesis-project02

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing
```bash
python src/preprocessing/clean_data.py
python src/preprocessing/feature_engineering.py
```

### 2. Run Classification Models
```bash
python src/classification/train_models.py
```

### 3. Perform Clustering Analysis
```bash
python src/clustering/cluster_analysis.py
```

### 4. Mine Association Rules
```bash
python src/association_rules/mine_rules.py
```

### 5. Generate Visualizations
```bash
python src/dimensionality_reduction/visualize.py
```

### 6. Run Statistical Analysis
```bash
python src/statistical_analysis/analyze.py
```

### 7. Launch Dashboard
```bash
streamlit run src/deployment/app.py
```

## Dashboard Features
- 📊 Upload CSV for instant predictions
- 🎯 Classification results (species, susceptibility, MAR class)
- 🔍 Cluster assignment and visualization
- 🔗 Association rules and co-resistance patterns
- 📈 Interactive PCA/t-SNE/UMAP plots
- 📥 Downloadable analysis reports

## Docker Deployment
```bash
# Build image
docker build -t antibiotic-resistance-ml .

# Run container
docker run -p 8501:8501 antibiotic-resistance-ml
```

## Key Findings
(To be updated after analysis)
- Classification accuracy: TBD
- Top resistance patterns: TBD
- High-risk sites: TBD
- MDR prevalence: TBD

## Public Health Implications
This analysis supports:
- Antimicrobial stewardship programs
- Surveillance of drug-resistant bacteria
- Risk assessment for water and food sources
- Evidence-based treatment guidelines

## Contributors
- Thesis Project Team

## License
MIT License

## References
(To be added)

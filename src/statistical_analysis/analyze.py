"""
Statistical Pattern Recognition Module
Performs correlation analysis, hypothesis testing, and feature importance with SHAP
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, pearsonr
import shap
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Statistical analysis for antibiotic resistance patterns"""
    
    def __init__(self):
        """Initialize statistical analyzer"""
        self.data = None
        self.correlations = {}
        self.test_results = {}
        self.shap_values = {}
    
    def load_data(self, data_path):
        """Load feature-engineered data"""
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.data)} samples")
        return self
    
    def correlation_analysis(self):
        """Perform comprehensive correlation analysis"""
        logger.info("Performing correlation analysis...")
        
        # Get resistance columns
        resistance_cols = [col for col in self.data.columns if col.endswith('_resistant')]
        
        if not resistance_cols:
            logger.warning("No resistance columns found")
            return self
        
        # Antibiotic-to-antibiotic correlations
        logger.info("\n=== Antibiotic-to-Antibiotic Correlations ===")
        resistance_data = self.data[resistance_cols]
        
        # Calculate Pearson correlation for binary data
        self.correlations['antibiotic'] = resistance_data.corr(method='pearson')
        
        # Find highly correlated pairs
        corr_matrix = self.correlations['antibiotic']
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.6:  # Threshold for high correlation
                    high_corr_pairs.append({
                        'antibiotic_1': corr_matrix.columns[i],
                        'antibiotic_2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)
            logger.info(f"\nHighly correlated antibiotic pairs (|r| > 0.6):")
            logger.info(f"\n{high_corr_df.head(10)}")
        
        return self
    
    def species_resistance_correlation(self):
        """Analyze correlation between species and resistance patterns"""
        logger.info("\n=== Species-Resistance Correlations ===")
        
        if 'bacterial_species' not in self.data.columns:
            logger.warning("Species column not found")
            return self
        
        resistance_cols = [col for col in self.data.columns if col.endswith('_resistant')]
        
        # Calculate resistance rates by species
        species_resistance = self.data.groupby('bacterial_species')[resistance_cols].mean()
        
        self.correlations['species_resistance'] = species_resistance
        
        logger.info("\nResistance rates by species:")
        logger.info(f"\n{species_resistance}")
        
        # Perform chi-square tests
        chi_square_results = []
        for col in resistance_cols:
            contingency_table = pd.crosstab(
                self.data['bacterial_species'],
                self.data[col]
            )
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            chi_square_results.append({
                'antibiotic': col,
                'chi_square': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        chi_square_df = pd.DataFrame(chi_square_results).sort_values('p_value')
        self.test_results['species_resistance_chi2'] = chi_square_df
        
        logger.info("\nChi-square tests (species vs resistance):")
        logger.info(f"\n{chi_square_df.head(10)}")
        
        return self
    
    def site_resistance_correlation(self):
        """Analyze correlation between sampling sites and resistance"""
        logger.info("\n=== Site-Resistance Correlations ===")
        
        if 'sample_source' not in self.data.columns:
            logger.warning("Sample source column not found")
            return self
        
        resistance_cols = [col for col in self.data.columns if col.endswith('_resistant')]
        
        # Calculate resistance rates by site
        site_resistance = self.data.groupby('sample_source')[resistance_cols].mean()
        
        self.correlations['site_resistance'] = site_resistance
        
        logger.info("\nResistance rates by sample source:")
        logger.info(f"\n{site_resistance}")
        
        # Perform chi-square tests
        chi_square_results = []
        for col in resistance_cols:
            contingency_table = pd.crosstab(
                self.data['sample_source'],
                self.data[col]
            )
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            chi_square_results.append({
                'antibiotic': col,
                'chi_square': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        chi_square_df = pd.DataFrame(chi_square_results).sort_values('p_value')
        self.test_results['site_resistance_chi2'] = chi_square_df
        
        logger.info("\nChi-square tests (site vs resistance):")
        logger.info(f"\n{chi_square_df.head(10)}")
        
        return self
    
    def mar_resistance_correlation(self):
        """Analyze correlation between MAR index and antibiotic responses"""
        logger.info("\n=== MAR Index-Resistance Correlations ===")
        
        if 'mar_index' not in self.data.columns:
            logger.warning("MAR index not found")
            return self
        
        resistance_cols = [col for col in self.data.columns if col.endswith('_resistant')]
        
        # Calculate Spearman correlation (MAR is continuous, resistance is binary)
        mar_correlations = []
        
        for col in resistance_cols:
            corr, p_value = spearmanr(self.data['mar_index'], self.data[col])
            mar_correlations.append({
                'antibiotic': col,
                'spearman_r': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        mar_corr_df = pd.DataFrame(mar_correlations).sort_values('spearman_r', ascending=False)
        self.correlations['mar_resistance'] = mar_corr_df
        
        logger.info("\nMAR index correlations with resistance:")
        logger.info(f"\n{mar_corr_df.head(10)}")
        
        return self
    
    def hypothesis_testing(self):
        """Perform hypothesis tests for key relationships"""
        logger.info("\n=== Hypothesis Testing ===")
        
        # Test 1: Does MAR index differ by species?
        if 'bacterial_species' in self.data.columns and 'mar_index' in self.data.columns:
            logger.info("\nTest 1: MAR index differences between species")
            
            species_groups = [group['mar_index'].dropna() for name, group in self.data.groupby('bacterial_species')]
            
            if len(species_groups) >= 2:
                # Kruskal-Wallis test (non-parametric)
                h_stat, p_value = stats.kruskal(*species_groups)
                
                self.test_results['mar_by_species'] = {
                    'test': 'Kruskal-Wallis',
                    'h_statistic': h_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                logger.info(f"  Kruskal-Wallis H: {h_stat:.4f}, p-value: {p_value:.4f}")
                logger.info(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} difference in MAR by species")
        
        # Test 2: Does MDR prevalence differ by sample source?
        if 'sample_source' in self.data.columns and 'mdr_category' in self.data.columns:
            logger.info("\nTest 2: MDR prevalence by sample source")
            
            contingency_table = pd.crosstab(
                self.data['sample_source'],
                self.data['mdr_category']
            )
            
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            self.test_results['mdr_by_source'] = {
                'test': 'Chi-square',
                'chi2': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'significant': p_value < 0.05
            }
            
            logger.info(f"  Chi-square: {chi2:.4f}, p-value: {p_value:.4f}")
            logger.info(f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} association")
        
        # Test 3: ESBL vs resistance patterns
        if 'esbl_positive' in self.data.columns:
            logger.info("\nTest 3: ESBL association with resistance")
            
            resistance_cols = [col for col in self.data.columns if col.endswith('_resistant')]
            esbl_associations = []
            
            for col in resistance_cols[:5]:  # Test first 5 antibiotics as example
                contingency_table = pd.crosstab(
                    self.data['esbl_positive'],
                    self.data[col]
                )
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                esbl_associations.append({
                    'antibiotic': col,
                    'chi2': chi2,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
            
            esbl_df = pd.DataFrame(esbl_associations)
            self.test_results['esbl_associations'] = esbl_df
            
            logger.info(f"\n{esbl_df}")
        
        return self
    
    def calculate_shap_values(self, model_path, sample_size=100):
        """
        Calculate SHAP values for feature importance
        
        NOTE: This method is currently disabled in the main() function due to 
        stability issues causing segmentation faults with certain model types.
        Consider using LIME or other interpretability methods as alternatives.
        
        Args:
            model_path: Path to trained model
            sample_size: Number of samples for SHAP analysis
        """
        logger.info(f"\nCalculating SHAP values from {model_path}...")
        
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Prepare features (use same features as model training)
            feature_cols = []
            feature_cols.extend([col for col in self.data.columns if col.endswith('_encoded') 
                               and not col.startswith('bacterial_species')])
            feature_cols.extend([col for col in self.data.columns if col.endswith('_resistant')])
            feature_cols.extend([col for col in self.data.columns if col.endswith('_mic_log')])
            
            derived_cols = ['mdr_score', 'mdr_percentage', 'esbl_positive']
            feature_cols.extend([col for col in derived_cols if col in self.data.columns])
            
            feature_cols = [col for col in feature_cols if col in self.data.columns]
            
            X = self.data[feature_cols].fillna(0).head(sample_size)
            
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                # Use a smaller sample size to avoid memory issues
                X_sample = X.sample(min(50, len(X)), random_state=42)
                
                explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
                shap_values = explainer.shap_values(X_sample, check_additivity=False)
                
                # For multi-class, take mean across classes
                if isinstance(shap_values, list):
                    shap_values = np.abs(shap_values).mean(axis=0)
                
                self.shap_values['model'] = {
                    'shap_values': shap_values,
                    'feature_names': feature_cols,
                    'X': X_sample
                }
                
                # Calculate mean absolute SHAP values
                mean_shap = np.abs(shap_values).mean(axis=0)
                shap_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': mean_shap
                }).sort_values('importance', ascending=False)
                
                logger.info("\nTop 10 features by SHAP importance:")
                logger.info(f"\n{shap_importance.head(10)}")
                
                self.correlations['shap_importance'] = shap_importance
            else:
                logger.warning("Model doesn't support SHAP analysis")
        
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
        
        return self
    
    def create_correlation_heatmap(self, output_path):
        """Create heatmap of antibiotic correlations"""
        logger.info("Creating correlation heatmap...")
        
        if 'antibiotic' not in self.correlations:
            logger.warning("No antibiotic correlations available")
            return self
        
        corr_matrix = self.correlations['antibiotic']
        
        # Shorten column/row names
        corr_matrix.columns = [col.replace('_resistant', '').replace('_', ' ')[:20] for col in corr_matrix.columns]
        corr_matrix.index = corr_matrix.columns
        
        # Create heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        plt.title('Antibiotic Co-Resistance Correlation Matrix', fontsize=14, pad=20)
        plt.tight_layout()
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap saved to {output_path}")
        return self
    
    def create_species_resistance_heatmap(self, output_path):
        """Create heatmap of species vs resistance rates"""
        logger.info("Creating species-resistance heatmap...")
        
        if 'species_resistance' not in self.correlations:
            logger.warning("No species-resistance correlations available")
            return self
        
        data = self.correlations['species_resistance']
        
        # Shorten column names
        data.columns = [col.replace('_resistant', '').replace('_', ' ')[:20] for col in data.columns]
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            data,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Resistance Rate'}
        )
        plt.title('Resistance Rates by Bacterial Species', fontsize=14, pad=20)
        plt.xlabel('Antibiotic')
        plt.ylabel('Species')
        plt.tight_layout()
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap saved to {output_path}")
        return self
    
    def save_results(self, output_dir):
        """Save statistical analysis results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving statistical analysis results to {output_dir}")
        
        # Save correlations
        for name, data in self.correlations.items():
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_dir / f'{name}_correlation.csv')
        
        # Save test results
        for name, data in self.test_results.items():
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_dir / f'{name}_tests.csv', index=False)
            elif isinstance(data, dict):
                pd.DataFrame([data]).to_csv(output_dir / f'{name}_test.csv', index=False)
        
        logger.info("Statistical analysis results saved successfully")
        return self


def main():
    """Main execution function"""
    # Paths
    base_path = Path(__file__).parent.parent.parent
    features_path = base_path / 'data' / 'processed' / 'features.csv'
    results_dir = base_path / 'data' / 'results' / 'statistical_analysis'
    
    # Find best model for SHAP analysis
    models_dir = base_path / 'data' / 'results'
    model_files = list(models_dir.glob('best_model_*.pkl'))
    model_path = model_files[0] if model_files else None
    
    # Perform statistical analysis
    analyzer = StatisticalAnalyzer()
    (analyzer.load_data(features_path)
             .correlation_analysis()
             .species_resistance_correlation()
             .site_resistance_correlation()
             .mar_resistance_correlation()
             .hypothesis_testing()
             .create_correlation_heatmap(results_dir / 'antibiotic_correlation_heatmap.png')
             .create_species_resistance_heatmap(results_dir / 'species_resistance_heatmap.png')
             .save_results(results_dir))
    
    # Calculate SHAP values if model available (disabled due to stability issues)
    # if model_path:
    #     analyzer.calculate_shap_values(model_path, sample_size=100)
    
    logger.info("\n✓ Statistical analysis completed successfully!")


if __name__ == "__main__":
    main()

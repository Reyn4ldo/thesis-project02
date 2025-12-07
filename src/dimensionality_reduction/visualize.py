"""
Dimensionality Reduction Module
Implements PCA, t-SNE, and UMAP for visualization of resistance patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """Reduce dimensionality and visualize resistance patterns"""
    
    def __init__(self):
        """Initialize dimensionality reducer"""
        self.data = None
        self.X = None
        self.X_scaled = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.models = {}
        self.embeddings = {}
    
    def load_data(self, data_path):
        """Load feature-engineered data"""
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.data)} samples")
        return self
    
    def prepare_features(self):
        """Prepare features for dimensionality reduction"""
        logger.info("Preparing features...")
        
        # Select resistance-related features
        feature_cols = []
        
        # Binary resistance flags
        feature_cols.extend([col for col in self.data.columns if col.endswith('_resistant')])
        
        # MIC log values
        feature_cols.extend([col for col in self.data.columns if col.endswith('_mic_log')])
        
        # MDR-related features
        derived_cols = ['mdr_score', 'mdr_percentage', 'esbl_positive', 'mar_index']
        feature_cols.extend([col for col in derived_cols if col in self.data.columns])
        
        self.feature_columns = [col for col in feature_cols if col in self.data.columns]
        
        # Create feature matrix
        self.X = self.data[self.feature_columns].fillna(0)
        
        # Standardize features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        logger.info(f"Prepared {len(self.feature_columns)} features")
        logger.info(f"Original dimensionality: {self.X_scaled.shape}")
        
        return self
    
    def perform_pca(self, n_components=2):
        """
        Perform PCA for linear dimensionality reduction
        
        Args:
            n_components: Number of principal components
        """
        logger.info(f"Performing PCA with {n_components} components...")
        
        self.models['pca'] = PCA(n_components=n_components, random_state=42)
        self.embeddings['pca'] = self.models['pca'].fit_transform(self.X_scaled)
        
        # Log explained variance
        explained_var = self.models['pca'].explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        logger.info(f"Explained variance ratio: {explained_var}")
        logger.info(f"Cumulative variance: {cumulative_var}")
        logger.info(f"Total variance explained: {cumulative_var[-1]:.2%}")
        
        return self
    
    def perform_tsne(self, n_components=2, perplexity=30):
        """
        Perform t-SNE for non-linear dimensionality reduction
        
        Args:
            n_components: Number of dimensions
            perplexity: t-SNE perplexity parameter
        """
        logger.info(f"Performing t-SNE with {n_components} components (perplexity={perplexity})...")
        
        self.models['tsne'] = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            max_iter=1000
        )
        self.embeddings['tsne'] = self.models['tsne'].fit_transform(self.X_scaled)
        
        logger.info("t-SNE completed")
        return self
    
    def perform_umap(self, n_components=2, n_neighbors=15, min_dist=0.1):
        """
        Perform UMAP for topology-preserving dimensionality reduction
        
        Args:
            n_components: Number of dimensions
            n_neighbors: Size of local neighborhood
            min_dist: Minimum distance between points
        """
        logger.info(f"Performing UMAP with {n_components} components...")
        
        self.models['umap'] = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        self.embeddings['umap'] = self.models['umap'].fit_transform(self.X_scaled)
        
        logger.info("UMAP completed")
        return self
    
    def visualize_2d(self, method='pca', color_by='bacterial_species', output_path=None, title=None):
        """
        Create 2D visualization
        
        Args:
            method: 'pca', 'tsne', or 'umap'
            color_by: Column to color points by
            output_path: Path to save figure
            title: Plot title
        """
        if method not in self.embeddings:
            logger.warning(f"No embeddings found for {method}")
            return self
        
        logger.info(f"Creating 2D visualization for {method} colored by {color_by}...")
        
        embedding = self.embeddings[method]
        
        # Prepare color data
        if color_by in self.data.columns:
            color_data = self.data[color_by]
            
            # Handle categorical vs numerical data
            if pd.api.types.is_numeric_dtype(color_data):
                is_categorical = False
            else:
                is_categorical = True
                # Encode categorical data
                unique_vals = color_data.unique()
                color_mapping = {val: i for i, val in enumerate(unique_vals)}
                color_data = color_data.map(color_mapping)
        else:
            color_data = None
            is_categorical = False
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        if is_categorical and color_data is not None:
            # Scatter plot with discrete colors
            for val in np.unique(color_data):
                mask = color_data == val
                label = list(color_mapping.keys())[list(color_mapping.values()).index(val)]
                plt.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    label=label,
                    alpha=0.6,
                    s=50
                )
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Scatter plot with continuous colors
            scatter = plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=color_data if color_data is not None else 'blue',
                cmap='viridis',
                alpha=0.6,
                s=50
            )
            if color_data is not None:
                plt.colorbar(scatter, label=color_by)
        
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(title or f'{method.upper()} Visualization - Colored by {color_by}')
        plt.tight_layout()
        
        if output_path:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {output_path}")
        
        plt.close()
        return self
    
    def visualize_3d(self, method='pca', color_by='bacterial_species', output_path=None):
        """Create 3D visualization"""
        if method not in self.embeddings:
            logger.warning(f"No embeddings found for {method}")
            return self
        
        embedding = self.embeddings[method]
        
        if embedding.shape[1] < 3:
            logger.warning(f"Need 3D embeddings for 3D plot, got {embedding.shape[1]}D")
            return self
        
        logger.info(f"Creating 3D visualization for {method}...")
        
        # Prepare color data
        if color_by in self.data.columns:
            color_data = self.data[color_by]
            if not pd.api.types.is_numeric_dtype(color_data):
                unique_vals = color_data.unique()
                color_mapping = {val: i for i, val in enumerate(unique_vals)}
                color_data = color_data.map(color_mapping)
        else:
            color_data = None
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=color_data if color_data is not None else 'blue',
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_zlabel(f'{method.upper()} Component 3')
        ax.set_title(f'{method.upper()} 3D Visualization - Colored by {color_by}')
        
        if color_data is not None:
            plt.colorbar(scatter, label=color_by, ax=ax, pad=0.1)
        
        plt.tight_layout()
        
        if output_path:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {output_path}")
        
        plt.close()
        return self
    
    def plot_pca_variance(self, output_path=None):
        """Plot explained variance ratio for PCA"""
        if 'pca' not in self.models:
            logger.warning("PCA not performed yet")
            return self
        
        logger.info("Plotting PCA explained variance...")
        
        pca = self.models['pca']
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        ax1.bar(range(1, len(explained_var) + 1), explained_var)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Individual Explained Variance')
        
        # Cumulative variance
        ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker='o')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {output_path}")
        
        plt.close()
        return self
    
    def create_comparison_plot(self, color_by='bacterial_species', output_path=None):
        """Create side-by-side comparison of all methods"""
        logger.info("Creating comparison plot...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        methods = ['pca', 'tsne', 'umap']
        
        for idx, method in enumerate(methods):
            if method not in self.embeddings:
                continue
            
            embedding = self.embeddings[method]
            ax = axes[idx]
            
            # Prepare color data
            if color_by in self.data.columns:
                color_data = self.data[color_by]
                if not pd.api.types.is_numeric_dtype(color_data):
                    unique_vals = color_data.unique()
                    color_mapping = {val: i for i, val in enumerate(unique_vals)}
                    color_data = color_data.map(color_mapping)
            else:
                color_data = None
            
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=color_data if color_data is not None else 'blue',
                cmap='viridis',
                alpha=0.6,
                s=30
            )
            
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel(f'{method.upper()} 2')
            ax.set_title(f'{method.upper()}')
        
        plt.suptitle(f'Dimensionality Reduction Comparison - Colored by {color_by}', fontsize=14)
        plt.tight_layout()
        
        if output_path:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {output_path}")
        
        plt.close()
        return self
    
    def save_results(self, output_dir):
        """Save embeddings and models"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving dimensionality reduction results to {output_dir}")
        
        # Save models
        for method, model in self.models.items():
            if method != 'tsne':  # t-SNE can't transform new data
                joblib.dump(model, output_dir / f'{method}_model.pkl')
        
        # Save embeddings
        for method, embedding in self.embeddings.items():
            embedding_df = pd.DataFrame(
                embedding,
                columns=[f'{method}_dim_{i+1}' for i in range(embedding.shape[1])]
            )
            embedding_df.to_csv(output_dir / f'{method}_embeddings.csv', index=False)
        
        # Save data with embeddings
        full_data = self.data.copy()
        for method, embedding in self.embeddings.items():
            for i in range(embedding.shape[1]):
                full_data[f'{method}_dim_{i+1}'] = embedding[:, i]
        
        full_data.to_csv(output_dir / 'data_with_embeddings.csv', index=False)
        
        logger.info("Results saved successfully")
        return self


def main():
    """Main execution function"""
    # Paths
    base_path = Path(__file__).parent.parent.parent
    features_path = base_path / 'data' / 'processed' / 'features.csv'
    results_dir = base_path / 'data' / 'results' / 'dimensionality_reduction'
    
    # Perform dimensionality reduction
    reducer = DimensionalityReducer()
    reducer.load_data(features_path).prepare_features()
    
    # 2D reductions
    reducer.perform_pca(n_components=2)
    reducer.perform_tsne(n_components=2, perplexity=30)
    reducer.perform_umap(n_components=2)
    
    # Create visualizations colored by different attributes
    color_attributes = ['bacterial_species', 'sample_source', 'mdr_category', 'mar_class']
    
    for color_by in color_attributes:
        if color_by in reducer.data.columns:
            for method in ['pca', 'tsne', 'umap']:
                output_path = results_dir / f'{method}_2d_{color_by}.png'
                reducer.visualize_2d(method=method, color_by=color_by, output_path=output_path)
    
    # Create comparison plot
    reducer.create_comparison_plot(color_by='bacterial_species', 
                                   output_path=results_dir / 'comparison_species.png')
    
    # PCA variance plot
    reducer.plot_pca_variance(output_path=results_dir / 'pca_variance.png')
    
    # 3D visualizations
    reducer.perform_pca(n_components=3)
    reducer.visualize_3d(method='pca', color_by='bacterial_species', 
                        output_path=results_dir / 'pca_3d_species.png')
    
    # Save results
    reducer.save_results(results_dir)
    
    logger.info("\n✓ Dimensionality reduction completed successfully!")


if __name__ == "__main__":
    main()

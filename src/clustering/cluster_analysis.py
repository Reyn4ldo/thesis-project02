"""
Clustering Module
Implements k-Means, Hierarchical Clustering, and DBSCAN for MDR pattern detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResistanceClusterer:
    """Clustering analysis for antibiotic resistance patterns"""
    
    def __init__(self):
        """Initialize clusterer"""
        self.data = None
        self.X = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.models = {}
        self.labels = {}
        self.metrics = {}
    
    def load_data(self, data_path):
        """Load feature-engineered data"""
        logger.info(f"Loading data from {data_path}")
        self.data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.data)} samples")
        return self
    
    def prepare_features(self):
        """Prepare features for clustering"""
        logger.info("Preparing features for clustering...")
        
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
        
        logger.info(f"Prepared {len(self.feature_columns)} features for clustering")
        return self
    
    def perform_kmeans(self, n_clusters_range=(2, 10)):
        """
        Perform k-Means clustering with multiple k values
        
        Args:
            n_clusters_range: Range of k values to try
        """
        logger.info("Performing k-Means clustering...")
        
        best_k = None
        best_score = -1
        silhouette_scores = []
        
        for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            
            # Calculate silhouette score
            score = silhouette_score(self.X_scaled, labels)
            silhouette_scores.append(score)
            
            logger.info(f"k={k}: Silhouette Score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # Fit final model with best k
        logger.info(f"Best k: {best_k} (Silhouette: {best_score:.4f})")
        self.models['kmeans'] = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        self.labels['kmeans'] = self.models['kmeans'].fit_predict(self.X_scaled)
        
        # Calculate metrics
        self.metrics['kmeans'] = {
            'n_clusters': best_k,
            'silhouette': best_score,
            'davies_bouldin': davies_bouldin_score(self.X_scaled, self.labels['kmeans']),
            'calinski_harabasz': calinski_harabasz_score(self.X_scaled, self.labels['kmeans'])
        }
        
        return self
    
    def perform_hierarchical(self, n_clusters=5):
        """
        Perform Hierarchical Clustering
        
        Args:
            n_clusters: Number of clusters
        """
        logger.info("Performing Hierarchical Clustering...")
        
        # Fit hierarchical clustering
        self.models['hierarchical'] = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        self.labels['hierarchical'] = self.models['hierarchical'].fit_predict(self.X_scaled)
        
        # Calculate metrics
        self.metrics['hierarchical'] = {
            'n_clusters': n_clusters,
            'silhouette': silhouette_score(self.X_scaled, self.labels['hierarchical']),
            'davies_bouldin': davies_bouldin_score(self.X_scaled, self.labels['hierarchical']),
            'calinski_harabasz': calinski_harabasz_score(self.X_scaled, self.labels['hierarchical'])
        }
        
        logger.info(f"Hierarchical Clustering: {n_clusters} clusters, "
                   f"Silhouette: {self.metrics['hierarchical']['silhouette']:.4f}")
        
        return self
    
    def perform_dbscan(self, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering to detect MDR outliers
        
        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
        """
        logger.info("Performing DBSCAN...")
        
        # Fit DBSCAN
        self.models['dbscan'] = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels['dbscan'] = self.models['dbscan'].fit_predict(self.X_scaled)
        
        # Count clusters and outliers
        n_clusters = len(set(self.labels['dbscan'])) - (1 if -1 in self.labels['dbscan'] else 0)
        n_outliers = list(self.labels['dbscan']).count(-1)
        
        logger.info(f"DBSCAN: {n_clusters} clusters, {n_outliers} outliers")
        
        # Calculate silhouette only if we have multiple clusters
        if n_clusters > 1:
            # Filter out outliers for silhouette calculation
            mask = self.labels['dbscan'] != -1
            if mask.sum() > 1:
                silhouette = silhouette_score(self.X_scaled[mask], self.labels['dbscan'][mask])
            else:
                silhouette = None
        else:
            silhouette = None
        
        self.metrics['dbscan'] = {
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'silhouette': silhouette
        }
        
        return self
    
    def analyze_clusters(self):
        """Analyze cluster characteristics"""
        logger.info("\nAnalyzing cluster characteristics...")
        
        self.cluster_profiles = {}
        
        for method in ['kmeans', 'hierarchical', 'dbscan']:
            if method not in self.labels:
                continue
            
            logger.info(f"\n=== {method.upper()} Cluster Analysis ===")
            
            # Add cluster labels to data
            self.data[f'{method}_cluster'] = self.labels[method]
            
            profiles = []
            
            for cluster_id in sorted(self.data[f'{method}_cluster'].unique()):
                cluster_data = self.data[self.data[f'{method}_cluster'] == cluster_id]
                
                profile = {
                    'cluster': cluster_id,
                    'size': len(cluster_data),
                    'percentage': 100 * len(cluster_data) / len(self.data),
                    'mean_mdr_score': cluster_data['mdr_score'].mean() if 'mdr_score' in cluster_data.columns else None,
                    'mean_mar_index': cluster_data['mar_index'].mean() if 'mar_index' in cluster_data.columns else None,
                    'esbl_positive_rate': cluster_data['esbl_positive'].mean() if 'esbl_positive' in cluster_data.columns else None,
                }
                
                # Most common species
                if 'bacterial_species' in cluster_data.columns:
                    profile['dominant_species'] = cluster_data['bacterial_species'].mode()[0] if len(cluster_data['bacterial_species'].mode()) > 0 else 'unknown'
                
                # Most common source
                if 'sample_source' in cluster_data.columns:
                    profile['dominant_source'] = cluster_data['sample_source'].mode()[0] if len(cluster_data['sample_source'].mode()) > 0 else 'unknown'
                
                profiles.append(profile)
                
                # Log profile
                logger.info(f"\nCluster {cluster_id}:")
                logger.info(f"  Size: {profile['size']} ({profile['percentage']:.1f}%)")
                if profile['mean_mdr_score']:
                    logger.info(f"  Mean MDR Score: {profile['mean_mdr_score']:.2f}")
                if profile['mean_mar_index']:
                    logger.info(f"  Mean MAR Index: {profile['mean_mar_index']:.4f}")
                if profile['esbl_positive_rate']:
                    logger.info(f"  ESBL+ Rate: {profile['esbl_positive_rate']*100:.1f}%")
                if 'dominant_species' in profile:
                    logger.info(f"  Dominant Species: {profile['dominant_species']}")
                if 'dominant_source' in profile:
                    logger.info(f"  Dominant Source: {profile['dominant_source']}")
            
            self.cluster_profiles[method] = pd.DataFrame(profiles)
        
        return self
    
    def identify_mdr_clusters(self):
        """Identify high-risk MDR clusters"""
        logger.info("\nIdentifying MDR clusters...")
        
        if 'mdr_score' not in self.data.columns:
            logger.warning("MDR score not available")
            return self
        
        # Define MDR threshold (e.g., resistant to 3+ antibiotics)
        mdr_threshold = 3
        
        for method in ['kmeans', 'hierarchical']:
            if f'{method}_cluster' not in self.data.columns:
                continue
            
            cluster_mdr_rates = self.data.groupby(f'{method}_cluster').apply(
                lambda x: (x['mdr_score'] >= mdr_threshold).mean()
            )
            
            logger.info(f"\n{method.upper()} - MDR prevalence by cluster:")
            for cluster_id, mdr_rate in cluster_mdr_rates.items():
                status = "HIGH-RISK" if mdr_rate > 0.5 else "LOW-RISK"
                logger.info(f"  Cluster {cluster_id}: {mdr_rate*100:.1f}% MDR [{status}]")
        
        return self
    
    def visualize_dendrogram(self, output_path, max_samples=100):
        """Create dendrogram for hierarchical clustering"""
        logger.info("Creating dendrogram...")
        
        # Sample data if too large
        if len(self.X_scaled) > max_samples:
            indices = np.random.choice(len(self.X_scaled), max_samples, replace=False)
            X_sample = self.X_scaled[indices]
        else:
            X_sample = self.X_scaled
        
        # Calculate linkage
        Z = linkage(X_sample, method='ward')
        
        # Plot dendrogram
        plt.figure(figsize=(12, 6))
        dendrogram(Z)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.tight_layout()
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dendrogram saved to {output_path}")
        return self
    
    def create_heatmap(self, output_path, method='kmeans'):
        """Create cluster heatmap"""
        logger.info(f"Creating heatmap for {method}...")
        
        if f'{method}_cluster' not in self.data.columns:
            logger.warning(f"Cluster labels for {method} not available")
            return self
        
        # Get resistance columns
        resistance_cols = [col for col in self.data.columns if col.endswith('_resistant')]
        
        if not resistance_cols:
            logger.warning("No resistance columns found")
            return self
        
        # Calculate resistance rates per cluster
        cluster_resistance = self.data.groupby(f'{method}_cluster')[resistance_cols].mean()
        
        # Shorten column names for visualization
        cluster_resistance.columns = [col.replace('_resistant', '').replace('_', ' ') for col in cluster_resistance.columns]
        
        # Plot heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(cluster_resistance, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Resistance Rate'})
        plt.title(f'{method.capitalize()} Cluster Resistance Patterns')
        plt.xlabel('Antibiotic')
        plt.ylabel('Cluster')
        plt.tight_layout()
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap saved to {output_path}")
        return self
    
    def save_results(self, output_dir):
        """Save clustering results and models"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving clustering results to {output_dir}")
        
        # Save models
        for method, model in self.models.items():
            joblib.dump(model, output_dir / f'{method}_model.pkl')
        
        # Save cluster labels
        labels_df = pd.DataFrame(self.labels)
        labels_df.to_csv(output_dir / 'cluster_labels.csv', index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.to_csv(output_dir / 'cluster_metrics.csv')
        
        # Save cluster profiles
        for method, profiles in self.cluster_profiles.items():
            profiles.to_csv(output_dir / f'{method}_profiles.csv', index=False)
        
        # Save data with cluster assignments
        self.data.to_csv(output_dir / 'data_with_clusters.csv', index=False)
        
        logger.info("Clustering results saved successfully")
        return self


def main():
    """Main execution function"""
    # Paths
    base_path = Path(__file__).parent.parent.parent
    features_path = base_path / 'data' / 'processed' / 'features.csv'
    results_dir = base_path / 'data' / 'results' / 'clustering'
    
    # Perform clustering
    clusterer = ResistanceClusterer()
    (clusterer.load_data(features_path)
              .prepare_features()
              .perform_kmeans(n_clusters_range=(2, 8))
              .perform_hierarchical(n_clusters=5)
              .perform_dbscan(eps=0.5, min_samples=5)
              .analyze_clusters()
              .identify_mdr_clusters()
              .visualize_dendrogram(results_dir / 'dendrogram.png')
              .create_heatmap(results_dir / 'kmeans_heatmap.png', method='kmeans')
              .create_heatmap(results_dir / 'hierarchical_heatmap.png', method='hierarchical')
              .save_results(results_dir))
    
    logger.info("\n✓ Clustering analysis completed successfully!")


if __name__ == "__main__":
    main()

"""
Streamlit Dashboard for Antibiotic Resistance Pattern Recognition
Provides interactive interface for analysis and predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Antibiotic Resistance Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


class AntibioticResistanceDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        """Initialize dashboard"""
        self.base_path = Path(__file__).parent.parent.parent
        self.data_path = self.base_path / 'data' / 'processed' / 'features.csv'
        self.results_path = self.base_path / 'data' / 'results'
        
        # Load data if available
        self.data = None
        if self.data_path.exists():
            self.data = pd.read_csv(self.data_path)
    
    def render_home(self):
        """Render home page"""
        st.markdown('<h1 class="main-header">🔬 Antibiotic Resistance Pattern Recognition System</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        ### Welcome to the Antibiotic Resistance Analysis Platform
        
        This comprehensive system analyzes antibiotic resistance patterns in bacterial isolates using
        advanced machine learning and statistical methods.
        
        #### 📊 Available Analysis Modules:
        
        1. **📈 Data Overview** - Explore the dataset and view summary statistics
        2. **🎯 Classification** - Predict species, susceptibility, and MAR class
        3. **🔍 Clustering** - Identify MDR clusters and resistance patterns
        4. **🔗 Association Rules** - Discover co-resistance patterns
        5. **📉 Dimensionality Reduction** - Visualize hidden structures (PCA, t-SNE, UMAP)
        6. **📊 Statistical Analysis** - Correlation analysis and hypothesis testing
        7. **⚡ Quick Prediction** - Upload CSV for instant analysis
        
        #### 🎯 Key Features:
        - **Multi-pattern recognition**: Classification, clustering, association mining
        - **Comprehensive visualizations**: Interactive plots and heatmaps
        - **Statistical insights**: Correlation analysis and hypothesis testing
        - **Downloadable reports**: Export results in CSV format
        - **Real-time predictions**: Upload data for instant analysis
        
        #### 📝 Dataset Information:
        """)
        
        if self.data is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", len(self.data))
            
            with col2:
                st.metric("Bacterial Species", self.data['bacterial_species'].nunique() if 'bacterial_species' in self.data.columns else "N/A")
            
            with col3:
                st.metric("Sample Sources", self.data['sample_source'].nunique() if 'sample_source' in self.data.columns else "N/A")
            
            with col4:
                st.metric("Antibiotics Tested", len([col for col in self.data.columns if col.endswith('_resistant')]))
        
        st.info("👈 Use the sidebar to navigate between different analysis modules")
    
    def render_data_overview(self):
        """Render data overview page"""
        st.markdown('<h2 class="sub-header">📈 Data Overview</h2>', unsafe_allow_html=True)
        
        if self.data is None:
            st.error("Data not found. Please run preprocessing first.")
            return
        
        # Dataset info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total Samples:** {len(self.data)}")
            st.write(f"**Total Features:** {len(self.data.columns)}")
        
        with col2:
            if 'mdr_score' in self.data.columns:
                st.write(f"**Mean MDR Score:** {self.data['mdr_score'].mean():.2f}")
                st.write(f"**Mean MAR Index:** {self.data['mar_index'].mean():.4f}" if 'mar_index' in self.data.columns else "")
        
        # Sample data
        st.subheader("Sample Data")
        st.dataframe(self.data.head(20))
        
        # Species distribution
        if 'bacterial_species' in self.data.columns:
            st.subheader("Species Distribution")
            species_counts = self.data['bacterial_species'].value_counts()
            fig = px.bar(x=species_counts.index, y=species_counts.values,
                        labels={'x': 'Species', 'y': 'Count'},
                        title='Distribution of Bacterial Species')
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample source distribution
        if 'sample_source' in self.data.columns:
            st.subheader("Sample Source Distribution")
            source_counts = self.data['sample_source'].value_counts()
            fig = px.pie(values=source_counts.values, names=source_counts.index,
                        title='Sample Source Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # MDR distribution
        if 'mdr_category' in self.data.columns:
            st.subheader("MDR Category Distribution")
            mdr_counts = self.data['mdr_category'].value_counts()
            fig = px.bar(x=mdr_counts.index, y=mdr_counts.values,
                        labels={'x': 'MDR Category', 'y': 'Count'},
                        title='Multi-Drug Resistance Categories',
                        color=mdr_counts.index,
                        color_discrete_map={'non-mdr': 'green', 'mdr': 'orange', 'high-mdr': 'red'})
            st.plotly_chart(fig, use_container_width=True)
    
    def render_classification(self):
        """Render classification results"""
        st.markdown('<h2 class="sub-header">🎯 Classification Results</h2>', unsafe_allow_html=True)
        
        # Load classification results
        results_files = list(self.results_path.glob('classification_results_*.csv'))
        
        if not results_files:
            st.warning("No classification results found. Please run classification first.")
            return
        
        # Select task
        task = st.selectbox("Select Classification Task", 
                           ["species", "mar_class", "susceptibility"])
        
        results_file = self.results_path / f'classification_results_{task}.csv'
        
        if results_file.exists():
            results = pd.read_csv(results_file)
            
            # Display metrics
            st.subheader(f"Performance Metrics - {task.replace('_', ' ').title()}")
            
            test_results = results[results['dataset'] == 'test']
            
            col1, col2, col3, col4 = st.columns(4)
            
            for idx, (_, row) in enumerate(test_results.iterrows()):
                if idx >= 4:
                    break
                
                with [col1, col2, col3, col4][idx]:
                    st.metric(row['model'], f"{row['accuracy']:.3f}", 
                             help=f"F1: {row['f1_macro']:.3f}")
            
            # Detailed results table
            st.subheader("Detailed Results")
            st.dataframe(test_results[['model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc']])
            
            # Feature importance
            feature_imp_file = self.results_path / f'feature_importance_{task}.csv'
            if feature_imp_file.exists():
                st.subheader("Top Feature Importances")
                feature_imp = pd.read_csv(feature_imp_file)
                
                fig = px.bar(feature_imp.head(15), x='importance', y='feature',
                            orientation='h',
                            title='Top 15 Most Important Features',
                            labels={'importance': 'Importance', 'feature': 'Feature'})
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Results file not found for task: {task}")
    
    def render_clustering(self):
        """Render clustering results"""
        st.markdown('<h2 class="sub-header">🔍 Clustering Analysis</h2>', unsafe_allow_html=True)
        
        clustering_dir = self.results_path / 'clustering'
        
        if not clustering_dir.exists():
            st.warning("No clustering results found. Please run clustering first.")
            return
        
        # Load cluster data
        data_file = clustering_dir / 'data_with_clusters.csv'
        if data_file.exists():
            cluster_data = pd.read_csv(data_file)
            
            # Method selection
            method = st.selectbox("Select Clustering Method", ["kmeans", "hierarchical", "dbscan"])
            
            if f'{method}_cluster' in cluster_data.columns:
                st.subheader(f"{method.capitalize()} Clustering Results")
                
                # Cluster distribution
                cluster_counts = cluster_data[f'{method}_cluster'].value_counts().sort_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                                labels={'x': 'Cluster', 'y': 'Count'},
                                title='Cluster Size Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                                title='Cluster Proportions')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster profiles
                profiles_file = clustering_dir / f'{method}_profiles.csv'
                if profiles_file.exists():
                    st.subheader("Cluster Characteristics")
                    profiles = pd.read_csv(profiles_file)
                    st.dataframe(profiles)
                
                # Heatmap
                heatmap_file = clustering_dir / f'{method}_heatmap.png'
                if heatmap_file.exists():
                    st.subheader("Resistance Pattern Heatmap")
                    st.image(str(heatmap_file))
        
        # Dendrogram for hierarchical
        if method == 'hierarchical':
            dendrogram_file = clustering_dir / 'dendrogram.png'
            if dendrogram_file.exists():
                st.subheader("Hierarchical Clustering Dendrogram")
                st.image(str(dendrogram_file))
    
    def render_association_rules(self):
        """Render association rules"""
        st.markdown('<h2 class="sub-header">🔗 Association Rule Mining</h2>', unsafe_allow_html=True)
        
        assoc_dir = self.results_path / 'association_rules'
        
        if not assoc_dir.exists():
            st.warning("No association rules found. Please run association mining first.")
            return
        
        # Method selection
        method = st.selectbox("Select Algorithm", ["Apriori", "FP-Growth"])
        
        rules_file = assoc_dir / f'{method.lower()}_rules.csv'
        
        if rules_file.exists():
            rules = pd.read_csv(rules_file)
            
            st.subheader(f"{method} Association Rules")
            st.write(f"**Total Rules Found:** {len(rules)}")
            
            # Filter controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_conf = st.slider("Minimum Confidence", 0.0, 1.0, 0.6, 0.05)
            
            with col2:
                min_supp = st.slider("Minimum Support", 0.0, 1.0, 0.1, 0.05)
            
            with col3:
                min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.2, 0.1)
            
            # Filter rules
            filtered_rules = rules[
                (rules['confidence'] >= min_conf) &
                (rules['support'] >= min_supp) &
                (rules['lift'] >= min_lift)
            ].sort_values('confidence', ascending=False)
            
            st.write(f"**Filtered Rules:** {len(filtered_rules)}")
            
            # Display top rules
            st.subheader("Top Co-Resistance Rules")
            
            for idx, row in filtered_rules.head(20).iterrows():
                with st.expander(f"Rule {idx+1}: {row['antecedents']} → {row['consequents']}"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Confidence", f"{row['confidence']:.2%}")
                    col2.metric("Support", f"{row['support']:.2%}")
                    col3.metric("Lift", f"{row['lift']:.2f}")
            
            # Downloadable results
            st.download_button(
                label="📥 Download Rules (CSV)",
                data=filtered_rules.to_csv(index=False),
                file_name=f"{method.lower()}_filtered_rules.csv",
                mime="text/csv"
            )
        else:
            st.error(f"Rules file not found for {method}")
    
    def render_dimensionality_reduction(self):
        """Render dimensionality reduction visualizations"""
        st.markdown('<h2 class="sub-header">📉 Dimensionality Reduction</h2>', unsafe_allow_html=True)
        
        dimred_dir = self.results_path / 'dimensionality_reduction'
        
        if not dimred_dir.exists():
            st.warning("No dimensionality reduction results found. Please run visualization first.")
            return
        
        # Load data with embeddings
        data_file = dimred_dir / 'data_with_embeddings.csv'
        if not data_file.exists():
            st.error("Embeddings data not found")
            return
        
        embed_data = pd.read_csv(data_file)
        
        # Method selection
        col1, col2 = st.columns(2)
        
        with col1:
            method = st.selectbox("Select Method", ["PCA", "t-SNE", "UMAP"])
        
        with col2:
            color_by = st.selectbox("Color By", 
                                   ["bacterial_species", "sample_source", "mdr_category", "mar_class"])
        
        # Create visualization
        method_lower = method.lower().replace('-', '')
        
        if f'{method_lower}_dim_1' in embed_data.columns and f'{method_lower}_dim_2' in embed_data.columns:
            fig = px.scatter(
                embed_data,
                x=f'{method_lower}_dim_1',
                y=f'{method_lower}_dim_2',
                color=color_by,
                title=f'{method} Visualization - Colored by {color_by.replace("_", " ").title()}',
                labels={
                    f'{method_lower}_dim_1': f'{method} Component 1',
                    f'{method_lower}_dim_2': f'{method} Component 2'
                },
                hover_data=['bacterial_species', 'sample_source'] if 'bacterial_species' in embed_data.columns else None
            )
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            st.plotly_chart(fig, use_container_width=True)
        
        # Show comparison images
        st.subheader("Comparison of Methods")
        comparison_file = dimred_dir / f'comparison_{color_by}.png'
        if comparison_file.exists():
            st.image(str(comparison_file))
        
        # PCA variance plot
        if method == "PCA":
            variance_file = dimred_dir / 'pca_variance.png'
            if variance_file.exists():
                st.subheader("PCA Explained Variance")
                st.image(str(variance_file))
    
    def render_statistical_analysis(self):
        """Render statistical analysis results"""
        st.markdown('<h2 class="sub-header">📊 Statistical Analysis</h2>', unsafe_allow_html=True)
        
        stats_dir = self.results_path / 'statistical_analysis'
        
        if not stats_dir.exists():
            st.warning("No statistical analysis results found. Please run analysis first.")
            return
        
        # Correlation heatmap
        st.subheader("Antibiotic Co-Resistance Correlations")
        heatmap_file = stats_dir / 'antibiotic_correlation_heatmap.png'
        if heatmap_file.exists():
            st.image(str(heatmap_file))
        
        # Species-resistance heatmap
        st.subheader("Resistance Rates by Species")
        species_heatmap = stats_dir / 'species_resistance_heatmap.png'
        if species_heatmap.exists():
            st.image(str(species_heatmap))
        
        # Correlation data
        corr_file = stats_dir / 'antibiotic_correlation.csv'
        if corr_file.exists():
            st.subheader("Correlation Matrix Data")
            corr_data = pd.read_csv(corr_file, index_col=0)
            st.dataframe(corr_data)
        
        # Hypothesis test results
        test_files = list(stats_dir.glob('*_test*.csv'))
        if test_files:
            st.subheader("Hypothesis Test Results")
            for test_file in test_files:
                st.write(f"**{test_file.stem.replace('_', ' ').title()}**")
                test_data = pd.read_csv(test_file)
                st.dataframe(test_data)
    
    def render_quick_prediction(self):
        """Render quick prediction interface"""
        st.markdown('<h2 class="sub-header">⚡ Quick Prediction</h2>', unsafe_allow_html=True)
        
        st.info("Upload a CSV file with antibiotic resistance data for instant analysis")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load uploaded data
                user_data = pd.read_csv(uploaded_file)
                
                st.success(f"✓ File uploaded: {len(user_data)} samples")
                
                # Show data preview
                with st.expander("Preview Uploaded Data"):
                    st.dataframe(user_data.head())
                
                st.warning("⚠️ Note: Prediction functionality requires trained models. "
                          "Please ensure preprocessing and model training are complete.")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    def run(self):
        """Run the dashboard"""
        # Sidebar navigation
        st.sidebar.title("🧭 Navigation")
        
        pages = {
            "🏠 Home": self.render_home,
            "📈 Data Overview": self.render_data_overview,
            "🎯 Classification": self.render_classification,
            "🔍 Clustering": self.render_clustering,
            "🔗 Association Rules": self.render_association_rules,
            "📉 Dimensionality Reduction": self.render_dimensionality_reduction,
            "📊 Statistical Analysis": self.render_statistical_analysis,
            "⚡ Quick Prediction": self.render_quick_prediction
        }
        
        page = st.sidebar.radio("Select Page", list(pages.keys()))
        
        # Sidebar info
        st.sidebar.markdown("---")
        st.sidebar.info("""
        **About This System**
        
        Comprehensive pattern recognition system for analyzing antibiotic resistance
        in bacterial isolates.
        
        Developed for thesis research on antimicrobial resistance surveillance.
        """)
        
        # Render selected page
        pages[page]()


def main():
    """Main application entry point"""
    dashboard = AntibioticResistanceDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()

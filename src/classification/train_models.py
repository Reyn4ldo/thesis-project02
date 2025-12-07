"""
Classification Module
Implements 6 supervised learning algorithms for antibiotic resistance prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AntibioticClassifier:
    """Multi-class classifier for antibiotic resistance patterns"""
    
    def __init__(self, task='species'):
        """
        Initialize classifier
        
        Args:
            task: Classification task ('species', 'susceptibility', or 'mar_class')
        """
        self.task = task
        self.models = {}
        self.results = {}
        self.feature_columns = None
        self.target_column = None
        
        # Define feature columns to use
        self.feature_types = {
            'encoded': True,      # Use S/I/R encoded values
            'resistant': True,    # Use binary resistance flags
            'mic_log': True,      # Use normalized MIC values
            'categorical': True,  # Use encoded categorical features
            'derived': True       # Use MDR scores and other derived features
        }
    
    def load_data(self, train_path, val_path, test_path):
        """Load train/validation/test splits"""
        logger.info("Loading data splits...")
        self.train_data = pd.read_csv(train_path)
        self.val_data = pd.read_csv(val_path)
        self.test_data = pd.read_csv(test_path)
        
        logger.info(f"Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")
        return self
    
    def prepare_features(self):
        """Prepare feature matrix and target vector"""
        logger.info(f"Preparing features for task: {self.task}")
        
        # Define target column based on task
        target_mapping = {
            'species': 'bacterial_species_encoded',
            'susceptibility': 'ampicillin_int',  # Can be changed to any antibiotic
            'mar_class': 'mar_class'
        }
        
        self.target_column = target_mapping.get(self.task)
        
        if self.target_column not in self.train_data.columns:
            raise ValueError(f"Target column {self.target_column} not found in data")
        
        # Select feature columns
        feature_cols = []
        
        if self.feature_types['encoded']:
            feature_cols.extend([col for col in self.train_data.columns if col.endswith('_encoded') 
                               and col != self.target_column])
        
        if self.feature_types['resistant']:
            feature_cols.extend([col for col in self.train_data.columns if col.endswith('_resistant')])
        
        if self.feature_types['mic_log']:
            feature_cols.extend([col for col in self.train_data.columns if col.endswith('_mic_log')])
        
        if self.feature_types['derived']:
            derived_cols = ['mdr_score', 'mdr_percentage', 'esbl_positive', 
                          'site_sample_count', 'source_sample_count']
            feature_cols.extend([col for col in derived_cols if col in self.train_data.columns])
        
        self.feature_columns = [col for col in feature_cols if col in self.train_data.columns]
        
        # Prepare datasets
        self.X_train = self.train_data[self.feature_columns].fillna(0)
        self.y_train = self.train_data[self.target_column]
        
        self.X_val = self.val_data[self.feature_columns].fillna(0)
        self.y_val = self.val_data[self.target_column]
        
        self.X_test = self.test_data[self.feature_columns].fillna(0)
        self.y_test = self.test_data[self.target_column]
        
        # For MAR class, encode string labels
        if self.task == 'mar_class':
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.y_train = self.label_encoder.fit_transform(self.y_train)
            self.y_val = self.label_encoder.transform(self.y_val)
            self.y_test = self.label_encoder.transform(self.y_test)
        
        # For susceptibility, encode S/I/R
        elif self.task == 'susceptibility':
            sir_mapping = {'S': 0, 'I': 1, 'R': 2}
            self.y_train = self.y_train.map(sir_mapping)
            self.y_val = self.y_val.map(sir_mapping)
            self.y_test = self.y_test.map(sir_mapping)
        
        logger.info(f"Features: {len(self.feature_columns)}")
        logger.info(f"Classes: {len(np.unique(self.y_train))}")
        
        return self
    
    def initialize_models(self):
        """Initialize all 6 classification models"""
        logger.info("Initializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'kNN': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }
        
        logger.info(f"Initialized {len(self.models)} models")
        return self
    
    def train_models(self):
        """Train all models"""
        logger.info("Training models...")
        
        self.trained_models = {}
        
        for name, model in self.models.items():
            logger.info(f"\nTraining {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                self.trained_models[name] = model
                logger.info(f"✓ {name} training complete")
            except Exception as e:
                logger.error(f"✗ Error training {name}: {str(e)}")
        
        return self
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        logger.info("\nEvaluating models...")
        
        self.results = {}
        
        for name, model in self.trained_models.items():
            logger.info(f"\nEvaluating {name}...")
            
            # Predictions
            y_pred_val = model.predict(self.X_val)
            y_pred_test = model.predict(self.X_test)
            
            # Get prediction probabilities for AUC
            try:
                y_proba_val = model.predict_proba(self.X_val)
                y_proba_test = model.predict_proba(self.X_test)
            except:
                y_proba_val = None
                y_proba_test = None
            
            # Calculate metrics
            metrics_val = self._calculate_metrics(self.y_val, y_pred_val, y_proba_val, 'validation')
            metrics_test = self._calculate_metrics(self.y_test, y_pred_test, y_proba_test, 'test')
            
            self.results[name] = {
                'validation': metrics_val,
                'test': metrics_test,
                'model': model
            }
            
            # Log results
            logger.info(f"Validation - Accuracy: {metrics_val['accuracy']:.4f}, F1: {metrics_val['f1_macro']:.4f}")
            logger.info(f"Test - Accuracy: {metrics_test['accuracy']:.4f}, F1: {metrics_test['f1_macro']:.4f}")
        
        return self
    
    def _calculate_metrics(self, y_true, y_pred, y_proba, dataset_name):
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Calculate AUC if probabilities available
        if y_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class AUC
                    y_true_bin = label_binarize(y_true, classes=range(n_classes))
                    metrics['roc_auc'] = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {str(e)}")
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
        
        return metrics
    
    def get_best_model(self):
        """Select best model based on F1 score"""
        logger.info("\nSelecting best model...")
        
        best_model_name = None
        best_f1 = 0
        
        for name, result in self.results.items():
            f1 = result['test']['f1_macro']
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
        
        logger.info(f"Best model: {best_model_name} (F1: {best_f1:.4f})")
        return best_model_name, self.results[best_model_name]['model']
    
    def get_feature_importance(self, model_name):
        """Extract feature importance from model"""
        logger.info(f"\nExtracting feature importance for {model_name}...")
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            logger.warning(f"Feature importance not available for {model_name}")
            return None
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 features:\n{feature_importance.head(10)}")
        
        return feature_importance
    
    def save_results(self, output_dir):
        """Save models and results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_dir}")
        
        # Save best model
        best_name, best_model = self.get_best_model()
        model_path = output_dir / f'best_model_{self.task}.pkl'
        joblib.dump(best_model, model_path)
        logger.info(f"Saved best model: {best_name}")
        
        # Save all results as CSV
        results_df = []
        for model_name, result in self.results.items():
            for dataset in ['validation', 'test']:
                row = {
                    'model': model_name,
                    'dataset': dataset,
                    'task': self.task,
                    **{k: v for k, v in result[dataset].items() if k != 'confusion_matrix'}
                }
                results_df.append(row)
        
        results_df = pd.DataFrame(results_df)
        results_df.to_csv(output_dir / f'classification_results_{self.task}.csv', index=False)
        
        # Save feature importance for best model
        feature_imp = self.get_feature_importance(best_name)
        if feature_imp is not None:
            feature_imp.to_csv(output_dir / f'feature_importance_{self.task}.csv', index=False)
        
        logger.info("Results saved successfully")
        return self


def main():
    """Main execution function"""
    # Paths
    base_path = Path(__file__).parent.parent.parent
    splits_dir = base_path / 'data' / 'processed' / 'splits'
    results_dir = base_path / 'data' / 'results'
    
    train_path = splits_dir / 'train.csv'
    val_path = splits_dir / 'val.csv'
    test_path = splits_dir / 'test.csv'
    
    # Train classifiers for each task
    tasks = ['species', 'mar_class']  # Add 'susceptibility' if needed
    
    for task in tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"CLASSIFICATION TASK: {task.upper()}")
        logger.info(f"{'='*60}")
        
        classifier = AntibioticClassifier(task=task)
        (classifier.load_data(train_path, val_path, test_path)
                   .prepare_features()
                   .initialize_models()
                   .train_models()
                   .evaluate_models()
                   .save_results(results_dir))
    
    logger.info("\n✓ All classification tasks completed successfully!")


if __name__ == "__main__":
    main()

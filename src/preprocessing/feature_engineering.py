"""
Feature Engineering Module
Converts antibiotic data to ML-friendly features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Transform cleaned data into ML-ready features"""
    
    def __init__(self, data_path):
        """
        Initialize FeatureEngineer
        
        Args:
            data_path: Path to cleaned CSV file
        """
        self.data_path = data_path
        self.data = None
        self.features_data = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load cleaned data"""
        logger.info(f"Loading cleaned data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        self.features_data = self.data.copy()
        logger.info(f"Loaded {len(self.data)} samples")
        return self
    
    def encode_sir_to_numerical(self):
        """
        Encode S/I/R interpretations to numerical values
        S=0 (susceptible), I=1 (intermediate), R=2 (resistant), U=-1 (unknown)
        """
        logger.info("Encoding S/I/R to numerical values...")
        
        int_columns = [col for col in self.data.columns if col.endswith('_int')]
        
        sir_mapping = {
            'S': 0,
            'I': 1,
            'R': 2,
            'U': -1
        }
        
        for col in int_columns:
            if col in self.features_data.columns:
                # Create numerical column
                new_col = col.replace('_int', '_encoded')
                self.features_data[new_col] = self.features_data[col].map(sir_mapping)
                
                # Fill any remaining NaN with -1 (unknown)
                self.features_data[new_col].fillna(-1, inplace=True)
        
        logger.info(f"Encoded {len(int_columns)} antibiotic interpretations")
        return self
    
    def create_binary_resistance_flags(self):
        """
        Create binary resistance flags (0=non-resistant, 1=resistant)
        Treats S and I as non-resistant, R as resistant
        """
        logger.info("Creating binary resistance flags...")
        
        int_columns = [col for col in self.data.columns if col.endswith('_int')]
        
        for col in int_columns:
            if col in self.features_data.columns:
                # Create binary column
                new_col = col.replace('_int', '_resistant')
                self.features_data[new_col] = (self.features_data[col] == 'R').astype(int)
        
        logger.info(f"Created {len(int_columns)} binary resistance flags")
        return self
    
    def normalize_mic_values(self):
        """Normalize MIC values to log scale and standardize"""
        logger.info("Normalizing MIC values...")
        
        mic_columns = [col for col in self.data.columns if col.endswith('_mic')]
        
        for col in mic_columns:
            if col in self.features_data.columns:
                # Extract numerical values from MIC strings (e.g., "≤2", "≥32", "4")
                mic_values = self.features_data[col].astype(str)
                
                # Clean MIC values
                mic_numeric = []
                for val in mic_values:
                    val = val.replace('≤', '').replace('≥', '').replace('*', '').strip()
                    try:
                        mic_numeric.append(float(val))
                    except:
                        mic_numeric.append(np.nan)
                
                # Create log-transformed column
                new_col = col.replace('_mic', '_mic_log')
                self.features_data[new_col] = np.log2(pd.Series(mic_numeric).replace(0, 0.001))
                
                # Fill NaN with median
                median_val = self.features_data[new_col].median()
                self.features_data[new_col].fillna(median_val, inplace=True)
        
        logger.info(f"Normalized {len(mic_columns)} MIC values")
        return self
    
    def create_mar_classes(self, thresholds=(0.15, 0.25)):
        """
        Create MAR classes: low (< 0.15), medium (0.15-0.25), high (> 0.25)
        
        Args:
            thresholds: Tuple of (low_threshold, high_threshold)
        """
        logger.info("Creating MAR classes...")
        
        if 'mar_index' in self.features_data.columns:
            low_thresh, high_thresh = thresholds
            
            def classify_mar(mar_value):
                if pd.isna(mar_value):
                    return 'unknown'
                elif mar_value < low_thresh:
                    return 'low'
                elif mar_value < high_thresh:
                    return 'medium'
                else:
                    return 'high'
            
            self.features_data['mar_class'] = self.features_data['mar_index'].apply(classify_mar)
            
            # Count distribution
            class_counts = self.features_data['mar_class'].value_counts()
            logger.info(f"MAR class distribution:\n{class_counts}")
        else:
            logger.warning("MAR index column not found")
        
        return self
    
    def calculate_mdr_score(self):
        """
        Calculate MDR (Multi-Drug Resistance) score
        Count number of resistant antibiotics per isolate
        """
        logger.info("Calculating MDR scores...")
        
        resistant_columns = [col for col in self.features_data.columns if col.endswith('_resistant')]
        
        if resistant_columns:
            self.features_data['mdr_score'] = self.features_data[resistant_columns].sum(axis=1)
            self.features_data['mdr_percentage'] = (
                self.features_data['mdr_score'] / len(resistant_columns) * 100
            )
            
            # Create MDR category (0-2: non-MDR, 3+: MDR, 6+: high-MDR)
            def classify_mdr(score):
                if score < 3:
                    return 'non-mdr'
                elif score < 6:
                    return 'mdr'
                else:
                    return 'high-mdr'
            
            self.features_data['mdr_category'] = self.features_data['mdr_score'].apply(classify_mdr)
            
            logger.info(f"MDR score statistics:")
            logger.info(f"  Mean: {self.features_data['mdr_score'].mean():.2f}")
            logger.info(f"  Median: {self.features_data['mdr_score'].median():.2f}")
            logger.info(f"  Max: {self.features_data['mdr_score'].max():.0f}")
            
            mdr_counts = self.features_data['mdr_category'].value_counts()
            logger.info(f"MDR category distribution:\n{mdr_counts}")
        
        return self
    
    def encode_categorical_features(self):
        """Encode categorical features using label encoding"""
        logger.info("Encoding categorical features...")
        
        categorical_cols = [
            'bacterial_species',
            'administrative_region',
            'national_site',
            'local_site',
            'sample_source',
            'esbl'
        ]
        
        for col in categorical_cols:
            if col in self.features_data.columns:
                # Create label encoder
                le = LabelEncoder()
                
                # Handle NaN values
                non_null_mask = self.features_data[col].notna()
                if non_null_mask.sum() > 0:
                    self.features_data.loc[non_null_mask, f'{col}_encoded'] = le.fit_transform(
                        self.features_data.loc[non_null_mask, col]
                    )
                    self.label_encoders[col] = le
                    
                    # Fill NaN with -1
                    self.features_data[f'{col}_encoded'].fillna(-1, inplace=True)
        
        logger.info(f"Encoded {len(categorical_cols)} categorical features")
        return self
    
    def create_regional_features(self):
        """Create features based on regional information"""
        logger.info("Creating regional features...")
        
        # Count samples per site
        if 'local_site' in self.features_data.columns:
            site_counts = self.features_data.groupby('local_site').size()
            self.features_data['site_sample_count'] = self.features_data['local_site'].map(site_counts)
        
        # Count samples per source type
        if 'sample_source' in self.features_data.columns:
            source_counts = self.features_data.groupby('sample_source').size()
            self.features_data['source_sample_count'] = self.features_data['sample_source'].map(source_counts)
        
        return self
    
    def create_esbl_features(self):
        """Create ESBL-related features"""
        logger.info("Creating ESBL features...")
        
        if 'esbl' in self.features_data.columns:
            # Binary ESBL indicator
            self.features_data['esbl_positive'] = (
                self.features_data['esbl'] == 'positive'
            ).astype(int)
            
            esbl_count = self.features_data['esbl_positive'].sum()
            logger.info(f"ESBL positive samples: {esbl_count} ({100*esbl_count/len(self.features_data):.1f}%)")
        
        return self
    
    def save_features(self, output_path):
        """Save engineered features to CSV"""
        logger.info(f"Saving engineered features to {output_path}")
        self.features_data.to_csv(output_path, index=False)
        logger.info("Features saved successfully")
        return self
    
    def create_train_test_split(self, output_dir, test_size=0.15, val_size=0.15, random_state=42):
        """
        Create train/validation/test splits (70/15/15)
        
        Args:
            output_dir: Directory to save splits
            test_size: Proportion for test set
            val_size: Proportion for validation set  
            random_state: Random seed for reproducibility
        """
        logger.info("Creating train/validation/test splits...")
        
        # Check if stratification is possible (no NaN values in bacterial_species)
        stratify_col = None
        if 'bacterial_species' in self.features_data.columns:
            if not self.features_data['bacterial_species'].isna().any():
                stratify_col = self.features_data['bacterial_species']
            else:
                logger.warning("bacterial_species contains NaN values, stratification disabled")
        
        # First split: separate test set
        train_val, test = train_test_split(
            self.features_data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
        stratify_col_val = None
        if stratify_col is not None:
            stratify_col_val = train_val['bacterial_species']
        
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_col_val
        )
        
        logger.info(f"Split sizes: Train={len(train)} ({100*len(train)/len(self.features_data):.1f}%), "
                   f"Val={len(val)} ({100*len(val)/len(self.features_data):.1f}%), "
                   f"Test={len(test)} ({100*len(test)/len(self.features_data):.1f}%)")
        
        # Save splits
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train.to_csv(output_dir / 'train.csv', index=False)
        val.to_csv(output_dir / 'val.csv', index=False)
        test.to_csv(output_dir / 'test.csv', index=False)
        
        logger.info("Splits saved successfully")
        return train, val, test
    
    def engineer_features(self, output_path, splits_dir):
        """
        Execute complete feature engineering pipeline
        
        Args:
            output_path: Path to save all engineered features
            splits_dir: Directory to save train/val/test splits
        """
        (self.load_data()
            .encode_sir_to_numerical()
            .create_binary_resistance_flags()
            .normalize_mic_values()
            .create_mar_classes()
            .calculate_mdr_score()
            .encode_categorical_features()
            .create_regional_features()
            .create_esbl_features()
            .save_features(output_path)
            .create_train_test_split(splits_dir))
        
        return self.features_data


def main():
    """Main execution function"""
    # Paths
    cleaned_data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'cleaned_data.csv'
    features_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'features.csv'
    splits_dir = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'splits'
    
    # Engineer features
    engineer = FeatureEngineer(cleaned_data_path)
    features = engineer.engineer_features(features_path, splits_dir)
    
    logger.info("\n✓ Feature engineering completed successfully!")
    logger.info(f"Total features created: {len(features.columns)}")


if __name__ == "__main__":
    main()

"""
Data Cleaning Module
Handles duplicate removal, missing values, and label standardization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and standardize antibiotic resistance data"""
    
    def __init__(self, data_path):
        """
        Initialize DataCleaner
        
        Args:
            data_path: Path to raw CSV file
        """
        self.data_path = data_path
        self.data = None
        self.cleaned_data = None
        
    def load_data(self):
        """Load raw data from CSV"""
        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns")
        return self
    
    def remove_duplicates(self):
        """Remove duplicate isolates based on key columns"""
        logger.info("Removing duplicate isolates...")
        initial_count = len(self.data)
        
        # Remove exact duplicates
        self.data = self.data.drop_duplicates()
        
        # Remove duplicates based on isolate_code (keeping first occurrence)
        self.data = self.data.drop_duplicates(subset=['isolate_code'], keep='first')
        
        removed = initial_count - len(self.data)
        logger.info(f"Removed {removed} duplicate rows. Remaining: {len(self.data)}")
        return self
    
    def standardize_labels(self):
        """Standardize categorical labels for S/I/R, species, and sites"""
        logger.info("Standardizing categorical labels...")
        
        # Standardize bacterial species names (lowercase with underscores)
        if 'bacterial_species' in self.data.columns:
            self.data['bacterial_species'] = self.data['bacterial_species'].str.lower().str.strip()
        
        # Standardize S/I/R labels across all _int columns
        int_columns = [col for col in self.data.columns if col.endswith('_int')]
        
        for col in int_columns:
            if col in self.data.columns:
                # Convert to uppercase and strip whitespace
                self.data[col] = self.data[col].astype(str).str.upper().str.strip()
                
                # Standardize variations
                self.data[col] = self.data[col].replace({
                    'S': 'S',
                    'I': 'I', 
                    'R': 'R',
                    '*R': 'R',  # Some entries have *R notation
                    '*I': 'I',
                    '*S': 'S',
                    'NAN': np.nan,
                    'NONE': np.nan,
                    '': np.nan
                })
        
        # Standardize site names
        for col in ['administrative_region', 'national_site', 'local_site', 'sample_source']:
            if col in self.data.columns:
                self.data[col] = self.data[col].str.lower().str.strip()
        
        # Standardize ESBL column
        if 'esbl' in self.data.columns:
            self.data['esbl'] = self.data['esbl'].astype(str).str.lower().str.strip()
            self.data['esbl'] = self.data['esbl'].replace({
                'neg': 'negative',
                'pos': 'positive',
                'c': 'unknown',
                'nan': np.nan,
                '': np.nan
            })
        
        logger.info("Label standardization complete")
        return self
    
    def handle_missing_values(self, strategy='mode'):
        """
        Handle missing antibiotic interpretation values
        
        Args:
            strategy: 'mode', 'unknown', or 'drop'
                - 'mode': Fill with most common value per antibiotic
                - 'unknown': Mark as 'unknown' category
                - 'drop': Drop rows with missing values
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        int_columns = [col for col in self.data.columns if col.endswith('_int')]
        
        if strategy == 'mode':
            # Fill missing values with mode (most common) for each antibiotic
            for col in int_columns:
                if col in self.data.columns:
                    mode_value = self.data[col].mode()
                    if len(mode_value) > 0:
                        self.data[col].fillna(mode_value[0], inplace=True)
                    else:
                        self.data[col].fillna('S', inplace=True)  # Default to susceptible
        
        elif strategy == 'unknown':
            # Mark missing as 'unknown'
            for col in int_columns:
                if col in self.data.columns:
                    self.data[col].fillna('U', inplace=True)
        
        elif strategy == 'drop':
            # Drop rows with any missing interpretation values
            self.data = self.data.dropna(subset=int_columns)
        
        logger.info(f"Missing value handling complete. Remaining rows: {len(self.data)}")
        return self
    
    def validate_data(self):
        """Validate data quality after cleaning"""
        logger.info("Validating cleaned data...")
        
        issues = []
        
        # Check for remaining duplicates in isolate_code
        if self.data['isolate_code'].duplicated().any():
            issues.append("Duplicate isolate codes found")
        
        # Check S/I/R labels are valid
        int_columns = [col for col in self.data.columns if col.endswith('_int')]
        valid_labels = {'S', 'I', 'R', 'U', np.nan}
        
        for col in int_columns:
            unique_vals = set(self.data[col].dropna().unique())
            invalid = unique_vals - valid_labels
            if invalid:
                issues.append(f"Invalid labels in {col}: {invalid}")
        
        if issues:
            logger.warning(f"Validation issues found: {issues}")
        else:
            logger.info("Data validation passed")
        
        return self
    
    def get_summary_statistics(self):
        """Generate summary statistics of the cleaned data"""
        logger.info("\n=== Data Summary Statistics ===")
        logger.info(f"Total samples: {len(self.data)}")
        logger.info(f"Bacterial species: {self.data['bacterial_species'].nunique()}")
        logger.info(f"Unique isolates: {self.data['isolate_code'].nunique()}")
        logger.info(f"Sample sources: {self.data['sample_source'].nunique()}")
        logger.info(f"Regions: {self.data['administrative_region'].nunique()}")
        
        # Count S/I/R distribution
        int_columns = [col for col in self.data.columns if col.endswith('_int')]
        logger.info(f"\nNumber of antibiotics tested: {len(int_columns)}")
        
        # Calculate resistance rates
        all_interpretations = []
        for col in int_columns:
            all_interpretations.extend(self.data[col].dropna().tolist())
        
        if all_interpretations:
            from collections import Counter
            counts = Counter(all_interpretations)
            total = sum(counts.values())
            logger.info("\nOverall interpretation distribution:")
            for label, count in counts.most_common():
                logger.info(f"  {label}: {count} ({100*count/total:.1f}%)")
        
        return self
    
    def save_cleaned_data(self, output_path):
        """Save cleaned data to CSV"""
        logger.info(f"Saving cleaned data to {output_path}")
        self.data.to_csv(output_path, index=False)
        logger.info("Data saved successfully")
        return self
    
    def clean(self, output_path, missing_strategy='mode'):
        """
        Execute complete cleaning pipeline
        
        Args:
            output_path: Path to save cleaned data
            missing_strategy: Strategy for handling missing values
        """
        (self.load_data()
            .remove_duplicates()
            .standardize_labels()
            .handle_missing_values(strategy=missing_strategy)
            .validate_data()
            .get_summary_statistics()
            .save_cleaned_data(output_path))
        
        self.cleaned_data = self.data
        return self.cleaned_data


def main():
    """Main execution function"""
    # Paths
    raw_data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'raw_data.csv'
    processed_data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'cleaned_data.csv'
    
    # Create processed directory if it doesn't exist
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean data
    cleaner = DataCleaner(raw_data_path)
    cleaned_data = cleaner.clean(processed_data_path, missing_strategy='mode')
    
    logger.info("\n✓ Data cleaning completed successfully!")


if __name__ == "__main__":
    main()

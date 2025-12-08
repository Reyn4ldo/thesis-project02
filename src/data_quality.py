"""
Data quality checking utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    Check data quality and provide warnings/recommendations
    """
    
    def __init__(self, df: pd.DataFrame, name: str = "Dataset"):
        """
        Initialize quality checker
        
        Args:
            df: DataFrame to check
            name: Name of the dataset for reporting
        """
        self.df = df
        self.name = name
        self.issues = []
        self.warnings = []
        self.info = []
        
    def check_all(self) -> Dict[str, Any]:
        """
        Run all quality checks
        
        Returns:
            Dictionary with quality report
        """
        self.check_missing_values()
        self.check_duplicates()
        self.check_data_types()
        self.check_outliers()
        self.check_distribution()
        self.check_column_names()
        
        return self.get_report()
    
    def check_missing_values(self) -> None:
        """Check for missing values"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        high_missing = missing_pct[missing_pct > 50]
        moderate_missing = missing_pct[(missing_pct > 10) & (missing_pct <= 50)]
        
        if len(high_missing) > 0:
            self.issues.append({
                'type': 'missing_values_high',
                'severity': 'high',
                'message': f"{len(high_missing)} column(s) have >50% missing values",
                'columns': list(high_missing.index),
                'details': {col: f"{pct:.1f}%" for col, pct in high_missing.items()}
            })
        
        if len(moderate_missing) > 0:
            self.warnings.append({
                'type': 'missing_values_moderate',
                'severity': 'medium',
                'message': f"{len(moderate_missing)} column(s) have 10-50% missing values",
                'columns': list(moderate_missing.index),
                'details': {col: f"{pct:.1f}%" for col, pct in moderate_missing.items()}
            })
        
        total_missing = missing.sum()
        if total_missing > 0:
            self.info.append({
                'type': 'missing_values_total',
                'message': f"Total missing values: {total_missing} ({(total_missing/(len(self.df)*len(self.df.columns)))*100:.2f}% of all data)"
            })
    
    def check_duplicates(self) -> None:
        """Check for duplicate rows"""
        n_duplicates = self.df.duplicated().sum()
        
        if n_duplicates > 0:
            duplicate_pct = (n_duplicates / len(self.df)) * 100
            
            if duplicate_pct > 5:
                self.issues.append({
                    'type': 'duplicates_high',
                    'severity': 'high',
                    'message': f"{n_duplicates} duplicate rows ({duplicate_pct:.1f}%)",
                    'recommendation': "Consider removing duplicates"
                })
            else:
                self.warnings.append({
                    'type': 'duplicates_low',
                    'severity': 'low',
                    'message': f"{n_duplicates} duplicate rows ({duplicate_pct:.1f}%)",
                    'recommendation': "Review and remove if necessary"
                })
    
    def check_data_types(self) -> None:
        """Check data types consistency"""
        type_issues = []
        
        for col in self.df.columns:
            # Check if numeric columns have non-numeric values
            if 'int' in col or 'mic' in col or 'score' in col or 'index' in col:
                if self.df[col].dtype == 'object':
                    type_issues.append({
                        'column': col,
                        'expected': 'numeric',
                        'actual': 'object/string'
                    })
        
        if type_issues:
            self.warnings.append({
                'type': 'data_type_mismatch',
                'severity': 'medium',
                'message': f"{len(type_issues)} columns may have incorrect data types",
                'details': type_issues
            })
    
    def check_outliers(self) -> None:
        """Check for outliers in numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_cols = []
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            n_outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            
            if n_outliers > 0:
                outlier_pct = (n_outliers / len(self.df)) * 100
                if outlier_pct > 5:
                    outlier_cols.append({
                        'column': col,
                        'count': n_outliers,
                        'percentage': f"{outlier_pct:.1f}%"
                    })
        
        if outlier_cols:
            self.info.append({
                'type': 'outliers_detected',
                'message': f"{len(outlier_cols)} column(s) have potential outliers",
                'details': outlier_cols,
                'note': "Outliers detected using 3*IQR method"
            })
    
    def check_distribution(self) -> None:
        """Check data distribution balance"""
        # Check categorical columns for imbalance
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if len(self.df[col].unique()) < 20:  # Only check if reasonable number of categories
                value_counts = self.df[col].value_counts()
                max_pct = (value_counts.iloc[0] / len(self.df)) * 100
                min_pct = (value_counts.iloc[-1] / len(self.df)) * 100
                
                if max_pct > 80:
                    self.warnings.append({
                        'type': 'severe_imbalance',
                        'severity': 'medium',
                        'message': f"Column '{col}' has severe class imbalance",
                        'details': f"Dominant class: {max_pct:.1f}%, Minority class: {min_pct:.1f}%",
                        'recommendation': "Consider class balancing techniques"
                    })
    
    def check_column_names(self) -> None:
        """Check column naming conventions"""
        issues = []
        
        for col in self.df.columns:
            # Check for spaces
            if ' ' in col:
                issues.append(f"'{col}' contains spaces")
            
            # Check for special characters (except underscore)
            if any(c in col for c in ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '{', '}', '[', ']']):
                issues.append(f"'{col}' contains special characters")
        
        if issues:
            self.info.append({
                'type': 'column_naming',
                'message': f"{len(issues)} column name(s) may need standardization",
                'details': issues[:5]  # Show first 5
            })
    
    def get_report(self) -> Dict[str, Any]:
        """
        Get comprehensive quality report
        
        Returns:
            Dictionary with quality assessment
        """
        return {
            'dataset_name': self.name,
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'issues': self.issues,
            'warnings': self.warnings,
            'info': self.info,
            'quality_score': self._calculate_quality_score()
        }
    
    def _calculate_quality_score(self) -> float:
        """
        Calculate overall quality score (0-100)
        
        Returns:
            Quality score
        """
        score = 100.0
        
        # Deduct points for issues
        score -= len(self.issues) * 15
        score -= len(self.warnings) * 5
        
        # Ensure score is between 0 and 100
        return max(0.0, min(100.0, score))
    
    def get_summary_text(self) -> str:
        """
        Get human-readable summary
        
        Returns:
            Summary text
        """
        report = self.get_report()
        
        lines = [
            f"Data Quality Report: {self.name}",
            f"={'='*50}",
            f"Rows: {report['total_rows']:,}",
            f"Columns: {report['total_columns']}",
            f"Quality Score: {report['quality_score']:.1f}/100",
            ""
        ]
        
        if report['issues']:
            lines.append("🚨 Critical Issues:")
            for issue in report['issues']:
                lines.append(f"  - {issue['message']}")
            lines.append("")
        
        if report['warnings']:
            lines.append("⚠️ Warnings:")
            for warning in report['warnings']:
                lines.append(f"  - {warning['message']}")
            lines.append("")
        
        if report['info']:
            lines.append("ℹ️ Information:")
            for info in report['info']:
                lines.append(f"  - {info['message']}")
        
        return "\n".join(lines)


def quick_quality_check(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
    """
    Quickly check data quality
    
    Args:
        df: DataFrame to check
        name: Dataset name
        
    Returns:
        Quality report dictionary
    """
    checker = DataQualityChecker(df, name)
    return checker.check_all()

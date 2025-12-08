"""
Export utilities for results in multiple formats
"""

import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultExporter:
    """
    Export analysis results to multiple formats (CSV, JSON, Excel)
    """
    
    def __init__(self, output_dir: str = 'exports'):
        """
        Initialize exporter
        
        Args:
            output_dir: Directory for exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_dataframe(self, df: pd.DataFrame, name: str, 
                        formats: List[str] = ['csv', 'json', 'excel']) -> Dict[str, Path]:
        """
        Export DataFrame to multiple formats
        
        Args:
            df: DataFrame to export
            name: Base name for files
            formats: List of formats ('csv', 'json', 'excel', 'html')
            
        Returns:
            Dictionary mapping format to file path
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exported_files = {}
        
        for fmt in formats:
            try:
                if fmt == 'csv':
                    file_path = self.output_dir / f"{name}_{timestamp}.csv"
                    df.to_csv(file_path, index=False)
                    exported_files['csv'] = file_path
                    logger.info(f"Exported CSV: {file_path}")
                    
                elif fmt == 'json':
                    file_path = self.output_dir / f"{name}_{timestamp}.json"
                    df.to_json(file_path, orient='records', indent=2)
                    exported_files['json'] = file_path
                    logger.info(f"Exported JSON: {file_path}")
                    
                elif fmt == 'excel':
                    file_path = self.output_dir / f"{name}_{timestamp}.xlsx"
                    df.to_excel(file_path, index=False, engine='openpyxl')
                    exported_files['excel'] = file_path
                    logger.info(f"Exported Excel: {file_path}")
                    
                elif fmt == 'html':
                    file_path = self.output_dir / f"{name}_{timestamp}.html"
                    df.to_html(file_path, index=False)
                    exported_files['html'] = file_path
                    logger.info(f"Exported HTML: {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to export {fmt}: {e}")
        
        return exported_files
    
    def export_dict(self, data: Dict[str, Any], name: str) -> Path:
        """
        Export dictionary to JSON
        
        Args:
            data: Dictionary to export
            name: Base name for file
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.output_dir / f"{name}_{timestamp}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Exported JSON: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to export dict: {e}")
            raise
    
    def export_metrics(self, metrics: Dict[str, float], name: str, 
                      description: str = "") -> Dict[str, Path]:
        """
        Export metrics with metadata
        
        Args:
            metrics: Dictionary of metric name -> value
            name: Base name for file
            description: Optional description
            
        Returns:
            Dictionary mapping format to file path
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create DataFrame
        df = pd.DataFrame([
            {'Metric': k, 'Value': v}
            for k, v in metrics.items()
        ])
        
        # Add metadata
        metadata = {
            'description': description,
            'timestamp': timestamp,
            'count': len(metrics)
        }
        
        exported_files = {}
        
        # Export CSV
        csv_path = self.output_dir / f"{name}_metrics_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        exported_files['csv'] = csv_path
        
        # Export JSON with metadata
        json_path = self.output_dir / f"{name}_metrics_{timestamp}.json"
        json_data = {
            'metadata': metadata,
            'metrics': metrics
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        exported_files['json'] = json_path
        
        logger.info(f"Exported metrics: {name}")
        return exported_files
    
    def export_summary(self, summary_dict: Dict[str, Any], name: str = "summary") -> Path:
        """
        Export analysis summary
        
        Args:
            summary_dict: Dictionary with summary information
            name: Base name for file
            
        Returns:
            Path to exported markdown file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.output_dir / f"{name}_{timestamp}.md"
        
        try:
            with open(file_path, 'w') as f:
                f.write(f"# Analysis Summary\n\n")
                f.write(f"**Generated:** {timestamp}\n\n")
                
                for section, content in summary_dict.items():
                    f.write(f"## {section}\n\n")
                    if isinstance(content, dict):
                        for key, value in content.items():
                            f.write(f"- **{key}:** {value}\n")
                    elif isinstance(content, list):
                        for item in content:
                            f.write(f"- {item}\n")
                    else:
                        f.write(f"{content}\n")
                    f.write("\n")
            
            logger.info(f"Exported summary: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to export summary: {e}")
            raise


class ModelExporter:
    """
    Export trained models with metadata
    """
    
    def __init__(self, output_dir: str = 'models'):
        """
        Initialize model exporter
        
        Args:
            output_dir: Directory for exported models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_model(self, model: Any, name: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
        """
        Export model with metadata
        
        Args:
            model: Trained model object
            name: Base name for files
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary mapping type to file path
        """
        import joblib
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exported_files = {}
        
        # Export model
        model_path = self.output_dir / f"{name}_{timestamp}.pkl"
        joblib.dump(model, model_path)
        exported_files['model'] = model_path
        logger.info(f"Exported model: {model_path}")
        
        # Export metadata if provided
        if metadata:
            meta_path = self.output_dir / f"{name}_{timestamp}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            exported_files['metadata'] = meta_path
            logger.info(f"Exported metadata: {meta_path}")
        
        return exported_files


def create_comparison_table(results: List[Dict[str, Any]], 
                           metrics: List[str]) -> pd.DataFrame:
    """
    Create comparison table for multiple models/methods
    
    Args:
        results: List of result dictionaries
        metrics: List of metric names to include
        
    Returns:
        DataFrame with comparison
    """
    comparison_data = []
    
    for result in results:
        row = {'Name': result.get('name', 'Unknown')}
        for metric in metrics:
            row[metric] = result.get(metric, None)
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

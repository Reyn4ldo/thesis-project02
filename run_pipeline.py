"""
Main Pipeline Script
Executes all analysis phases in sequence
"""

import sys
from pathlib import Path
import logging
import time
import traceback

# Setup logging
# Ensure log file can be created (current directory should exist, but being defensive)
log_dir = Path(__file__).parent
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))


def run_phase(phase_name, module_path, function_name='main'):
    """Run a pipeline phase"""
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING PHASE: {phase_name}")
    logger.info(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Import and run module
        module = __import__(module_path, fromlist=[function_name])
        main_func = getattr(module, function_name)
        
        main_func()
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Phase completed successfully in {elapsed:.2f} seconds")
        return True
    
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ Phase failed after {elapsed:.2f} seconds: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Execute complete analysis pipeline"""
    logger.info("="*80)
    logger.info("ANTIBIOTIC RESISTANCE PATTERN RECOGNITION PIPELINE")
    logger.info("="*80)
    
    pipeline_start = time.time()
    
    phases = [
        ("Phase 1: Data Cleaning", "preprocessing.clean_data"),
        ("Phase 2: Feature Engineering", "preprocessing.feature_engineering"),
        ("Phase 3: Classification", "classification.train_models"),
        ("Phase 4: Clustering", "clustering.cluster_analysis"),
        ("Phase 5: Association Rule Mining", "association_rules.mine_rules"),
        ("Phase 6: Dimensionality Reduction", "dimensionality_reduction.visualize"),
        ("Phase 7: Statistical Analysis", "statistical_analysis.analyze"),
    ]
    
    results = []
    
    for phase_name, module_path in phases:
        success = run_phase(phase_name, module_path)
        results.append((phase_name, success))
        
        if not success:
            logger.warning(f"Phase '{phase_name}' failed. Continuing with next phase...")
    
    # Summary
    pipeline_elapsed = time.time() - pipeline_start
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*80)
    
    for phase_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{status}: {phase_name}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    logger.info(f"\nTotal: {successful}/{total} phases completed successfully")
    logger.info(f"Total pipeline execution time: {pipeline_elapsed:.2f} seconds ({pipeline_elapsed/60:.2f} minutes)")
    
    if successful == total:
        logger.info("\n🎉 All phases completed successfully!")
        logger.info("\n📊 To view results, run: streamlit run src/deployment/app.py")
        sys.exit(0)
    else:
        logger.warning(f"\n⚠️ {total - successful} phase(s) failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

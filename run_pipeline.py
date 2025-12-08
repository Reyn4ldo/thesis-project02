"""
Main Pipeline Script
Executes all analysis phases in sequence with enhanced progress tracking
"""

import sys
from pathlib import Path
import logging
import time
import traceback
from typing import List, Tuple, Optional
from tqdm import tqdm

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


def run_phase(phase_name: str, module_path: str, function_name: str = 'main', 
              phase_num: int = 1, total_phases: int = 7) -> Tuple[bool, float, Optional[str]]:
    """
    Run a pipeline phase with enhanced error handling and timing
    
    Args:
        phase_name: Human-readable name of the phase
        module_path: Python module path to import
        function_name: Function name to call (default: 'main')
        phase_num: Current phase number for progress tracking
        total_phases: Total number of phases
        
    Returns:
        Tuple of (success: bool, elapsed_time: float, error_message: Optional[str])
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"[{phase_num}/{total_phases}] {phase_name}")
    logger.info(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Import and run module
        module = __import__(module_path, fromlist=[function_name])
        main_func = getattr(module, function_name)
        
        main_func()
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Phase completed successfully in {elapsed:.2f}s ({elapsed/60:.2f}m)")
        return True, elapsed, None
    
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"✗ Phase failed after {elapsed:.2f}s: {error_msg}")
        logger.debug(traceback.format_exc())
        return False, elapsed, error_msg


def main():
    """Execute complete analysis pipeline with enhanced progress tracking"""
    logger.info("="*80)
    logger.info("ANTIBIOTIC RESISTANCE PATTERN RECOGNITION PIPELINE")
    logger.info("Version 2.0 - Enhanced Edition")
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
    phase_times = []
    
    # Use tqdm for overall progress
    with tqdm(total=len(phases), desc="Overall Progress", unit="phase", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for idx, (phase_name, module_path) in enumerate(phases, 1):
            success, elapsed, error = run_phase(phase_name, module_path, 
                                               phase_num=idx, total_phases=len(phases))
            results.append((phase_name, success, error))
            phase_times.append(elapsed)
            pbar.update(1)
            
            if not success:
                logger.warning(f"⚠️ Phase '{phase_name}' failed. Continuing with next phase...")
    
    # Summary
    pipeline_elapsed = time.time() - pipeline_start
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*80)
    
    # Detailed results table
    logger.info(f"\n{'Phase':<50} {'Status':<15} {'Time (s)':<10}")
    logger.info("-" * 80)
    
    for (phase_name, success, error), phase_time in zip(results, phase_times):
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{phase_name:<50} {status:<15} {phase_time:>8.2f}s")
        if error:
            logger.info(f"  └─ Error: {error}")
    
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    # Statistics
    logger.info("\n" + "-" * 80)
    logger.info(f"Success Rate: {successful}/{total} phases ({successful/total*100:.1f}%)")
    logger.info(f"Total Time: {pipeline_elapsed:.2f}s ({pipeline_elapsed/60:.2f}m)")
    logger.info(f"Average Time per Phase: {sum(phase_times)/len(phase_times):.2f}s")
    logger.info(f"Fastest Phase: {min(phase_times):.2f}s")
    logger.info(f"Slowest Phase: {max(phase_times):.2f}s")
    
    if successful == total:
        logger.info("\n" + "="*80)
        logger.info("🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("\n📊 Next steps:")
        logger.info("  1. View results: streamlit run src/deployment/app.py")
        logger.info("  2. Generate report: python generate_report.py")
        logger.info("  3. Check outputs: data/results/")
        sys.exit(0)
    else:
        logger.warning("\n" + "="*80)
        logger.warning(f"⚠️ {total - successful} PHASE(S) FAILED")
        logger.warning("="*80)
        logger.warning("\n🔍 Troubleshooting steps:")
        logger.warning("  1. Check pipeline.log for detailed error messages")
        logger.warning("  2. Verify data files exist in data/raw/")
        logger.warning("  3. Ensure all dependencies are installed")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Utility functions for enhanced performance and functionality
"""

import functools
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Optional, Dict
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)


class ResultCache:
    """
    Simple file-based cache for expensive computations
    """
    
    def __init__(self, cache_dir: str = '.cache'):
        """
        Initialize cache
        
        Args:
            cache_dir: Directory to store cached results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate unique cache key from function name and arguments"""
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, func_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """
        Retrieve cached result if available
        
        Args:
            func_name: Name of the function
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Cached result or None if not found
        """
        cache_key = self._get_cache_key(func_name, args, kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    logger.debug(f"Cache hit for {func_name}")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None
        return None
    
    def set(self, func_name: str, args: tuple, kwargs: dict, result: Any) -> None:
        """
        Store result in cache
        
        Args:
            func_name: Name of the function
            args: Positional arguments
            kwargs: Keyword arguments
            result: Result to cache
        """
        cache_key = self._get_cache_key(func_name, args, kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            logger.debug(f"Cached result for {func_name}")
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")


# Global cache instance
_cache = ResultCache()


def cached(func: Callable) -> Callable:
    """
    Decorator to cache function results
    
    Usage:
        @cached
        def expensive_function(arg1, arg2):
            # expensive computation
            return result
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Try to get from cache
        result = _cache.get(func.__name__, args, kwargs)
        if result is not None:
            return result
        
        # Compute and cache
        result = func(*args, **kwargs)
        _cache.set(func.__name__, args, kwargs, result)
        return result
    
    return wrapper


def parallel_map(func: Callable, items: list, max_workers: int = None, 
                use_threads: bool = False, desc: str = "Processing") -> list:
    """
    Apply function to items in parallel with progress bar
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        max_workers: Maximum number of parallel workers (None = auto)
        use_threads: Use threads instead of processes
        desc: Description for progress bar
        
    Returns:
        List of results
    """
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    results = []
    with executor_class(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        
        for future in tqdm(futures, desc=desc, total=len(items)):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error in parallel execution: {e}")
                results.append(None)
    
    return results


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{int(minutes)}m {seconds % 60:.0f}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def format_bytes(bytes_size: int) -> str:
    """
    Format bytes into human-readable string
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if needed
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class Timer:
    """
    Context manager for timing code blocks
    
    Usage:
        with Timer("Processing data"):
            # expensive operation
            process_data()
    """
    
    def __init__(self, description: str = "Operation", logger_func=None):
        """
        Initialize timer
        
        Args:
            description: Description of the operation
            logger_func: Optional logging function (default: print)
        """
        self.description = description
        self.logger_func = logger_func or print
        self.start_time = None
        self.elapsed = None
        
    def __enter__(self):
        """Start timer"""
        self.start_time = time.time()
        self.logger_func(f"⏱️ Starting: {self.description}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log result"""
        self.elapsed = time.time() - self.start_time
        if exc_type is None:
            self.logger_func(f"✓ Completed: {self.description} in {format_time(self.elapsed)}")
        else:
            self.logger_func(f"✗ Failed: {self.description} after {format_time(self.elapsed)}")
        return False


def validate_dataframe(df, required_columns: list = None, 
                       min_rows: int = 1, name: str = "DataFrame") -> bool:
    """
    Validate DataFrame has required structure
    
    Args:
        df: Pandas DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        name: Name for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValueError(f"{name} must have at least {min_rows} rows, got {len(df)}")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
    
    return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default on division by zero
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return on division by zero
        
    Returns:
        Result of division or default
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

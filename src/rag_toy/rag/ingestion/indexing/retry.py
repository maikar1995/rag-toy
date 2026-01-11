"""Retry logic for Azure services with exponential backoff and jitter."""

import time
import random
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

BASE_DELAY = 1.0


def exponential_backoff_with_jitter(attempt: int, base_delay: float = BASE_DELAY, max_delay: float = 60.0) -> float:
    """
    Calculate delay with exponential backoff and jitter.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay in seconds with jitter applied
    """
    delay = min(max_delay, base_delay * (2 ** attempt))
    jitter = random.uniform(0, 0.25 * delay)
    return delay + jitter


def retry_with_backoff(func: Callable, max_retries: int, operation_name: str, *args, **kwargs) -> Any:
    """
    Execute function with retry logic and exponential backoff.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retries
        operation_name: Name for logging purposes
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        Result of successful function execution
        
    Raises:
        Exception: If all retries are exhausted
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            should_retry = False
            if hasattr(e, 'status_code'):
                status_code = e.status_code
                should_retry = status_code in [429, 500, 502, 503, 504]
            elif "timeout" in str(e).lower() or "connection" in str(e).lower():
                should_retry = True
            
            if attempt == max_retries or not should_retry:
                logger.error(f"❌ {operation_name} failed after {attempt + 1} attempts: {e}")
                raise e
                
            delay = exponential_backoff_with_jitter(attempt)
            logger.warning(f"⚠️  {operation_name} failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.2f}s: {e}")
            time.sleep(delay)
    
    raise Exception(f"{operation_name} exhausted all retries")
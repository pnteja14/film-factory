import time
from functools import wraps
from typing import Callable, TypeVar, Any
from utils.errors import RetryableException, PermanentException
from utils.logger import AgentLogger

T = TypeVar('T')

def retry_with_backoff(
    max_retries: int = 3,
    base_wait: float = 1.0,
    max_wait: float = 60.0,
    backoff_multiplier: float = 2.0,
    retryable_exceptions: tuple = (RetryableException,),
):
    """
    Decorator for retrying failed API calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_wait: Initial wait time in seconds
        max_wait: Maximum wait time between retries
        backoff_multiplier: Multiplier for exponential backoff
        retryable_exceptions: Tuple of exceptions that trigger retries
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            logger = AgentLogger(func.__name__)
            wait_time = base_wait
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except PermanentException as e:
                    logger.error(f"Permanent error, not retrying: {e}", e)
                    raise
                except retryable_exceptions as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries ({max_retries}) exceeded", e)
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    wait_time = min(wait_time * backoff_multiplier, max_wait)
            
        return wrapper
    return decorator
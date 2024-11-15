from functools import wraps
from typing import Callable
from time import time


def execute_interval(intervalSec: int):
    def decorator(func: Callable):
        last_called = [0.0]

        @wraps(func)
        def wrapped(*args, **kwargs):
            current_time = time()
            if current_time - last_called[0] >= intervalSec:
                last_called[0] = current_time
                return func(*args, **kwargs)

        return wrapped

    return decorator

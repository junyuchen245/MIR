"""Cache for sampling parameters intended for usecases where the same sampling
locations are used in a loop."""

from contextlib import AbstractContextManager
from threading import local
from typing import Any, Callable, List, Optional

_CACHE_VARIABLES = local()
_CACHE_VARIABLES.active_stack = []
_CACHE_VARIABLES.passive_stack = {}


def get_cached_sampling_parameters(func: Callable[[], Any]) -> Any:
    """Get sampling parameters from cache, if available, otherwise compute
    and cache them.

    If no cache is active, the function is called directly.
    """
    if _CACHE_VARIABLES.active_stack:
        cache = _CACHE_VARIABLES.active_stack[-1]
        if cache is not None:
            return cache.get_sampling_parameters(func)
    return func()


class sampling_cache(  # this is supposed to appear as function - pylint: disable=invalid-name
    AbstractContextManager
):
    """Context manager for enabling sampling parameter cache.

    Should surround a code block where the same sampling locations are used
    multiple times, e.g.:
    ```
    for training_step in range(1000):
        with sampler_cache():
            # code block with sampling operations
    ```

    This context can be used to store the sampling parameters for the current
    iteration and reuse them in the next iteration. If the same sampling
    parameters are used in multiple iterations, this can save some computation
    time, especially if CPU otherwise would be the bottleneck.

    Arguments:
        identifier: Optional identifier for the cache. If multiple caches are
            used, they have to be distinguished by the identifier (nesting
            caches is fine without an identifier).
    """

    def __init__(self, identifier: Optional[str] = None) -> None:
        self.identifier = identifier

    def __enter__(self) -> None:
        if _CACHE_VARIABLES.passive_stack.get(self.identifier, []):
            cache = _CACHE_VARIABLES.passive_stack[self.identifier].pop()
        else:
            cache = _SamplingParameterCache()
        _CACHE_VARIABLES.active_stack.append(cache)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        cache = _CACHE_VARIABLES.active_stack.pop()
        cache.reset()
        if not self.identifier in _CACHE_VARIABLES.passive_stack:
            _CACHE_VARIABLES.passive_stack[self.identifier] = []
        _CACHE_VARIABLES.passive_stack[self.identifier].append(cache)


class no_sampling_cache(  # this is supposed to appear as function - pylint: disable=invalid-name
    AbstractContextManager
):
    """Context manager for locally disabling sampling parameter cache."""

    def __enter__(self) -> None:
        _CACHE_VARIABLES.active_stack.append(None)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _CACHE_VARIABLES.active_stack.pop()


def clear_sampling_cache(identifier: Optional[str] = None) -> None:
    """Clear the cache for the given identifier.

    Active caches are not affected.
    """
    if identifier in _CACHE_VARIABLES.passive_stack:
        _CACHE_VARIABLES.passive_stack[identifier].clear()


class _SamplingParameterCache:
    """Cache for sampling parameters."""

    def __init__(self) -> None:
        self._sampling_parameters: List[Any] = []
        self._index: int = 0

    def get_sampling_parameters(self, func: Callable[[], Any]) -> Any:
        """Get sampling parameters from cache, if available, otherwise compute them."""
        if self._index < len(self._sampling_parameters):
            parameters = self._sampling_parameters[self._index]
        else:
            parameters = func()
            self._sampling_parameters.append(parameters)
        self._index += 1
        return parameters

    def reset(self) -> None:
        """Reset the cache."""
        self._index = 0

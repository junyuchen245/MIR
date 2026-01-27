"""Functionalities for setting the default sampler."""

from contextlib import ContextDecorator
from threading import local
from typing import Optional

from .interface import ISampler
from .interpolator import LinearInterpolator

_DEFAULT_SAMPLER: Optional[ISampler] = None
_DEFAULT_SAMPLER_CONTEXT_STACK = local()
_DEFAULT_SAMPLER_CONTEXT_STACK.stack = []


def get_default_sampler() -> ISampler:
    """Get current default sampler"""
    if _DEFAULT_SAMPLER_CONTEXT_STACK.stack:
        return _DEFAULT_SAMPLER_CONTEXT_STACK.stack[-1]
    if _DEFAULT_SAMPLER is None:
        return LinearInterpolator()
    return _DEFAULT_SAMPLER


def set_default_sampler(sampler: Optional[ISampler]) -> None:
    """Set default sampler

    Args:
        sampler: Sampler to set as default.
    """
    global _DEFAULT_SAMPLER  # pylint: disable=global-statement
    _DEFAULT_SAMPLER = sampler


def clear_default_sampler() -> None:
    """Clear default sampler"""
    global _DEFAULT_SAMPLER  # pylint: disable=global-statement
    _DEFAULT_SAMPLER = None


def get_sampler(sampler: Optional[ISampler]) -> ISampler:
    """Get sampler, either from argument or default"""
    return sampler if sampler is not None else get_default_sampler()


class default_sampler(  # this is supposed to appear as function - pylint: disable=invalid-name
    ContextDecorator
):
    """Context manager for setting default sampler

    Arguments:
        sampler: Sampler to set as default.
    """

    def __init__(self, sampler: ISampler):
        self.sampler = sampler

    def __enter__(self) -> None:
        _DEFAULT_SAMPLER_CONTEXT_STACK.stack.append(self.sampler)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _DEFAULT_SAMPLER_CONTEXT_STACK.stack.pop()

"""Interface definitions"""

from abc import abstractmethod
from typing import Protocol, Sequence

from torch import Tensor


class Interpolator(Protocol):
    """Interpolates values on regular grid in voxel coordinates"""

    @abstractmethod
    def __call__(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        """Interpolate

        Args:
            volume: Volume to be interpolated with shape
                (batch_size, *channel_dims, dim_1, ..., dim_{n_dims}). Dimension
                order is the same as the coordinate order of the coordinates
            coordinates: Interpolation coordinates with shape
                (batch_size, n_dims, *target_shape)

        Returns:
            Interpolated volume with shape (batch_size, *channel_dims, *target_shape)
        """


class FixedPointFunction(Protocol):
    """Protocol for fixed point functions"""

    def __call__(
        self,
        iteration_input: Tensor,
        output_buffer: Tensor,
    ) -> None:
        """Call a fixed point function

        Args:
            iteration_input: Input to the fixed point function
            output_buffer: Output should be stored in-place to this Tensor (same
                shape as the input)
        """


class FixedPointSolver(Protocol):
    """Protocol for fixed point solvers"""

    @abstractmethod
    def solve(
        self,
        fixed_point_function: FixedPointFunction,
        initial_value: Tensor,
    ) -> Tensor:
        """Solve a fixed point problem

        Args:
            fixed_point_function: Function to be iterated until convergence
            initial_value: Initial iteration value

        Returns:
            Solution of the fixed point iteration
        """


class FixedPointStopCriterion(Protocol):
    """Protocol for fixed point iteration stopping criterions"""

    @abstractmethod
    def should_stop(
        self,
        current_iteration: Tensor,
        previous_iterations: Sequence[Tensor],
        n_earlier_iterations: int,
    ) -> bool:
        """Return whether iterating should be continued at beginning of an iteration

        Args:
            current_iteration: Current output of the fixed point iteration. For
                n_earlier_iterations == 0 this equals the initial guess.
            previous_iterations: Previous outputs of the fixed point iteration,
                starting from the most recent one. length of this list may
                depend on the fixed point iteration solver and is always 0 for
                the n_earlier_iterations == 0.
            n_earlier_iterations: Number of calls made to the fixed point function
                before the next iteration.

        Returns:
            Whether the iteration should be stopped
        """

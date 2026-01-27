"""Fixed point iteration related implementations"""

from abc import abstractmethod
from typing import NamedTuple, Optional, Sequence

from torch import Tensor
from torch import abs as torch_abs
from torch import bmm, empty, eye
from torch import max as torch_max
from torch import zeros, zeros_like
from torch.linalg import solve

from .interface import FixedPointFunction, FixedPointSolver, FixedPointStopCriterion


class BaseCountingStopCriterion(FixedPointStopCriterion):
    """Base stop criterion definining min and max number of iterations

    Args:
        min_iterations: Minimum number of iterations to use
        max_iterations: Maximum number of iterations to use
        check_convergence_every_nth_iteration: Check convergence criterion
            only every nth iteration, does not have effect on stopping based on
            min or max number of iterations.
    """

    def __init__(
        self,
        min_iterations: int = 2,
        max_iterations: int = 50,
        check_convergence_every_nth_iteration: int = 1,
    ) -> None:
        self._min_iterations: int = min_iterations
        self._max_iterations = max_iterations
        self._check_convergence_every_nth_iteration = check_convergence_every_nth_iteration

    def _should_check_convergence(self, n_earlier_iterations: int) -> bool:
        return ((n_earlier_iterations + 1) % self._check_convergence_every_nth_iteration) == 0

    @abstractmethod
    def _should_stop(
        self,
        current_iteration: Tensor,
        previous_iterations: Sequence[Tensor],
        n_earlier_iterations: int,
    ) -> bool:
        """Return whether should stop before starting the iteration"""

    def should_stop(
        self,
        current_iteration: Tensor,
        previous_iterations: Sequence[Tensor],
        n_earlier_iterations: int,
    ) -> bool:
        if self._min_iterations > n_earlier_iterations:
            return False
        if self._max_iterations <= n_earlier_iterations:
            return True
        return self._should_check_convergence(n_earlier_iterations) and self._should_stop(
            current_iteration=current_iteration,
            previous_iterations=previous_iterations,
            n_earlier_iterations=n_earlier_iterations,
        )


class FixedIterationCountStopCriterion(BaseCountingStopCriterion):
    """Iteration is terminated based on fixed iteration count"""

    def __init__(
        self,
        n_iterations: int = 50,
    ) -> None:
        super().__init__(
            min_iterations=0,
            max_iterations=n_iterations,
        )

    def _should_stop(
        self,
        current_iteration: Tensor,
        previous_iterations: Sequence[Tensor],
        n_earlier_iterations: int,
    ) -> bool:
        return False


class MaxElementWiseAbsStopCriterion(BaseCountingStopCriterion):
    """Stops when no element-wise difference is larger than a threshold"""

    def __init__(
        self,
        min_iterations: int = 1,
        max_iterations: int = 50,
        threshold: float = 1e-2,
        check_convergence_every_nth_iteration: int = 1,
    ) -> None:
        super().__init__(
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            check_convergence_every_nth_iteration=check_convergence_every_nth_iteration,
        )
        self._threshold = threshold

    def _should_stop(
        self,
        current_iteration: Tensor,
        previous_iterations: Sequence[Tensor],
        n_earlier_iterations: int,
    ) -> bool:
        if len(previous_iterations) == 0:
            return False
        previous_iteration = previous_iterations[0]
        max_difference = torch_max(torch_abs(previous_iteration - current_iteration))
        return bool(max_difference < self._threshold)


class RelativeL2ErrorStopCriterion(BaseCountingStopCriterion):
    """Stops when relative L^2 error is below the set threshold"""

    def __init__(
        self,
        min_iterations: int = 1,
        max_iterations: int = 50,
        threshold: float = 1e-2,
        epsilon: float = 1e-5,
        check_convergence_every_nth_iteration: int = 1,
    ) -> None:
        super().__init__(
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            check_convergence_every_nth_iteration=check_convergence_every_nth_iteration,
        )
        self._threshold = threshold
        self._epsilon = epsilon

    def _should_stop(
        self,
        current_iteration: Tensor,
        previous_iterations: Sequence[Tensor],
        n_earlier_iterations: int,
    ) -> bool:
        if len(previous_iterations) == 0:
            return False
        previous_iteration = previous_iterations[0]
        error = (current_iteration - previous_iteration).norm() / (
            self._epsilon + current_iteration.norm().item()
        )
        return bool(error < self._threshold)


class AndersonSolverArguments(NamedTuple):
    """Arguments for Anderson solver

    Attributes:
        memory_length: How many iterations to store in memory
        beta: Beta value of Anderson fixed point solver
        matrix_epsilon: Epsilon value for avoiding division by zero
    """

    memory_length: int = 4
    beta: float = 1.0
    matrix_epsilon: float = 1e-4


class AndersonSolver(FixedPointSolver):
    """Anderson fixed point solver

    The implementation is based on code from the NeurIPS 2020 tutorial by Zico
    Kolter, David Duvenaud, and Matt Johnson.
    (http://implicit-layers-tutorial.org/deep_equilibrium_models/)

    Walker, Homer F., and Peng Ni. "Anderson acceleration for fixed-point iterations."
    SIAM Journal on Numerical Analysis 49.4 (2011): 1715-1735.

    Args:
        stop_criterion: Stop criterion to use
        arguments: Arguments for Anderson solver
    """

    def __init__(
        self,
        stop_criterion: Optional[FixedPointStopCriterion] = None,
        arguments: Optional[AndersonSolverArguments] = None,
    ) -> None:
        self._stop_criterion = (
            MaxElementWiseAbsStopCriterion() if stop_criterion is None else stop_criterion
        )
        self._arguments = AndersonSolverArguments() if arguments is None else arguments

    def solve(
        self,
        fixed_point_function: FixedPointFunction,
        initial_value: Tensor,
    ) -> Tensor:
        if self._stop_criterion.should_stop(
            current_iteration=initial_value, previous_iterations=[], n_earlier_iterations=0
        ):
            return initial_value
        initial_value = initial_value.detach()
        batch_size = initial_value.size(0)
        data_shape = initial_value.shape[1:]
        input_memory = zeros(
            (batch_size, self._arguments.memory_length) + data_shape,
            dtype=initial_value.dtype,
            device=initial_value.device,
        )
        output_memory = zeros_like(input_memory)
        input_memory[:, 0] = initial_value
        fixed_point_function(initial_value, output_memory[:, 0])
        if self._stop_criterion.should_stop(
            current_iteration=output_memory[:, 0],
            previous_iterations=list(input_memory[:, :1]),
            n_earlier_iterations=1,
        ):
            return output_memory[:, 0].clone()
        input_memory[:, 1] = output_memory[:, 0]
        fixed_point_function(output_memory[:, 0], output_memory[:, 1])
        coefficients_matrix = zeros(
            batch_size,
            self._arguments.memory_length + 1,
            self._arguments.memory_length + 1,
            dtype=initial_value.dtype,
            device=initial_value.device,
        )
        coefficients_matrix[:, 0, 1:] = coefficients_matrix[:, 1:, 0] = 1
        solving_target = zeros(
            batch_size,
            self._arguments.memory_length + 1,
            1,
            dtype=initial_value.dtype,
            device=initial_value.device,
        )
        solving_target[:, 0] = 1
        n_earlier_iterations = 2
        current_memory_length = min(n_earlier_iterations, self._arguments.memory_length)
        while not self._stop_criterion.should_stop(
            current_iteration=output_memory[
                :, (n_earlier_iterations - 1) % self._arguments.memory_length
            ],
            previous_iterations=[
                input_memory[
                    :, (n_earlier_iterations - 1 - memory_index) % self._arguments.memory_length
                ]
                for memory_index in range(current_memory_length)
            ],
            n_earlier_iterations=n_earlier_iterations,
        ):
            step_differences = (
                output_memory[:, :current_memory_length] - input_memory[:, :current_memory_length]
            ).view(batch_size, current_memory_length, -1)
            coefficients_matrix[:, 1 : current_memory_length + 1, 1 : current_memory_length + 1] = (
                bmm(step_differences, step_differences.transpose(1, 2))
                + self._arguments.matrix_epsilon
                * eye(
                    current_memory_length,
                    dtype=initial_value.dtype,
                    device=initial_value.device,
                )[None]
            )
            del step_differences
            alpha = solve(  # PyTorch bug - pylint: disable=not-callable
                coefficients_matrix[:, : current_memory_length + 1, : current_memory_length + 1],
                solving_target[:, : current_memory_length + 1],
            )[:, 1 : current_memory_length + 1, 0]
            input_memory[:, n_earlier_iterations % self._arguments.memory_length] = (
                self._arguments.beta
                * (
                    alpha[:, None]
                    @ output_memory[:, :current_memory_length].view(
                        batch_size, current_memory_length, -1
                    )
                )[:, 0]
            ).view_as(initial_value)
            if self._arguments.beta != 1.0:
                input_memory[:, n_earlier_iterations % self._arguments.memory_length] += (
                    (1 - self._arguments.beta)
                    * (
                        alpha[:, None]
                        @ input_memory[:, :current_memory_length].view(
                            batch_size, current_memory_length, -1
                        )
                    )[:, 0]
                ).view_as(initial_value)
            del alpha
            fixed_point_function(
                input_memory[:, n_earlier_iterations % self._arguments.memory_length],
                output_memory[:, n_earlier_iterations % self._arguments.memory_length],
            )
            n_earlier_iterations += 1
            current_memory_length = min(n_earlier_iterations, self._arguments.memory_length)
        return output_memory[:, (n_earlier_iterations - 1) % self._arguments.memory_length].clone()


class NaiveSolver(FixedPointSolver):
    """Naive fixed point solver

    Args:
        stop_criterion: Stop criterion to use
    """

    def __init__(
        self,
        stop_criterion: Optional[FixedPointStopCriterion] = None,
    ) -> None:
        self._stop_criterion = (
            MaxElementWiseAbsStopCriterion() if stop_criterion is None else stop_criterion
        )

    def solve(
        self,
        fixed_point_function: FixedPointFunction,
        initial_value: Tensor,
    ) -> Tensor:
        cache = empty(
            (2,) + initial_value.shape,
            dtype=initial_value.dtype,
            device=initial_value.device,
        )
        cache[0] = initial_value
        n_earlier_iterations = 0
        while not self._stop_criterion.should_stop(
            current_iteration=cache[n_earlier_iterations % 2],
            previous_iterations=[cache[(n_earlier_iterations + 1) % 2]]
            if n_earlier_iterations > 0
            else [],
            n_earlier_iterations=n_earlier_iterations,
        ):
            fixed_point_function(
                cache[n_earlier_iterations % 2], cache[(n_earlier_iterations + 1) % 2]
            )
            n_earlier_iterations += 1
        return cache[n_earlier_iterations % 2].clone()


class EmptySolver(FixedPointSolver):
    """Empty fixed point solver which returns the initial guess"""

    def solve(
        self,
        fixed_point_function: FixedPointFunction,
        initial_value: Tensor,
    ) -> Tensor:
        return initial_value

"""Grid sample implementation that allows for second order differentiation.

Copyright (c) 2022-2023 Aliaksandr Siarohin

Licensed under the MIT License

Slighly modified from:

https://github.com/AliaksandrSiarohin/cuda-gridsample-grad2
"""

import pathlib
from os.path import join

import torch
from torch.utils.cpp_extension import load

_CURRENT_DIRECTORY = pathlib.Path(__file__).parent.resolve()


gridsample_grad2 = None


def grid_sample_2d(input, grid, padding_mode="zeros", align_corners=True):
    global gridsample_grad2
    if gridsample_grad2 is None:
        # We load the extension dynamically to avoid unnecessary compilation (and
        # dependency on ninja).
        gridsample_grad2 = load(
            name="gridsample_grad2",
            sources=[
                join(_CURRENT_DIRECTORY, "gridsample_cuda.cpp"),
                join(_CURRENT_DIRECTORY, "gridsample_cuda.cu"),
            ],
            verbose=True,
        )
    assert padding_mode in ["zeros", "border"]
    return _GridSample2dForward.apply(input, grid, padding_mode, align_corners)


def grid_sample_3d(input, grid, padding_mode="zeros", align_corners=True):
    global gridsample_grad2
    if gridsample_grad2 is None:
        # We load the extension dynamically to avoid unnecessary compilation (and
        # dependency on ninja).
        gridsample_grad2 = load(
            name="gridsample_grad2",
            sources=[
                join(_CURRENT_DIRECTORY, "gridsample_cuda.cpp"),
                join(_CURRENT_DIRECTORY, "gridsample_cuda.cu"),
            ],
            verbose=True,
        )
    assert padding_mode in ["zeros", "border"]
    return _GridSample3dForward.apply(input, grid, padding_mode, align_corners)


class _GridSample2dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode=0, align_corners=True):
        assert input.ndim == 4
        assert grid.ndim == 4
        assert input.shape[0] == grid.shape[0]
        assert grid.shape[3] == 2

        output = torch.nn.functional.grid_sample(
            input=input,
            grid=grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        ctx.save_for_backward(input, grid)
        ctx.padding_mode = ["zeros", "border"].index(padding_mode)
        ctx.align_corners = align_corners
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSample2dBackward.apply(
            grad_output, input, grid, ctx.padding_mode, ctx.align_corners
        )
        return grad_input, grad_grid, None, None


class _GridSample2dBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid, padding_mode=0, align_corners=True):
        output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])
        grad_input, grad_grid = torch.ops.aten.grid_sampler_2d_backward(
            grad_output, input, grid, 0, padding_mode, align_corners, output_mask
        )

        ctx.save_for_backward(grad_output, input, grid)
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners

        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        grad_output, input, grid = ctx.saved_tensors
        assert (
            grad_output.is_cuda
            and input.is_cuda
            and grid.is_cuda
            and grad2_grad_input.is_cuda
            and grad2_grad_grid.is_cuda
        )
        out = gridsample_grad2.grad2_2d(
            grad2_grad_input,
            grad2_grad_grid,
            grad_output,
            input,
            grid,
            ctx.padding_mode,
            ctx.align_corners,
        )

        grad_grad_output = out[0]
        grad_input = out[1]
        grad_grid = out[2]

        return grad_grad_output, grad_input, grad_grid, None, None


class _GridSample3dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode=0, align_corners=True):
        assert input.ndim == 5
        assert grid.ndim == 5
        assert input.shape[0] == grid.shape[0]
        assert grid.shape[4] == 3

        output = torch.nn.functional.grid_sample(
            input=input,
            grid=grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        ctx.save_for_backward(input, grid)
        ctx.padding_mode = ["zeros", "border"].index(padding_mode)
        ctx.align_corners = align_corners
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSample3dBackward.apply(
            grad_output, input, grid, ctx.padding_mode, ctx.align_corners
        )
        return grad_input, grad_grid, None, None


class _GridSample3dBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid, padding_mode=0, align_corners=True):
        output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])
        grad_input, grad_grid = torch.ops.aten.grid_sampler_3d_backward(
            grad_output, input, grid, 0, padding_mode, align_corners, output_mask
        )

        ctx.save_for_backward(grad_output, input, grid)
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners

        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        grad_output, input, grid = ctx.saved_tensors
        assert (
            grad_output.is_cuda
            and input.is_cuda
            and grid.is_cuda
            and grad2_grad_input.is_cuda
            and grad2_grad_grid.is_cuda
        )
        out = gridsample_grad2.grad2_3d(
            grad2_grad_input,
            grad2_grad_grid,
            grad_output,
            input,
            grid,
            ctx.padding_mode,
            ctx.align_corners,
        )

        grad_grad_output = out[0]
        grad_input = out[1]
        grad_grid = out[2]

        return grad_grad_output, grad_input, grad_grid, None, None

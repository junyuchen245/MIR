"""Core matrix operations for generic affine transformations."""

from typing import Sequence

from torch import Tensor, allclose, cat, diag_embed, eye, inverse, matmul, ones, zeros

from MIR.models.SITReg.composable_mapping.util import (
    broadcast_tensors_in_parts,
    broadcast_to_in_parts,
    get_channel_dims,
    get_channels_shape,
    move_channels_first,
    move_channels_last,
    split_shape,
)

IDENTITY_MATRIX_TOLERANCE = 1e-5
ZERO_MATRIX_TOLERANCE = 1e-5


def compose_affine_matrices(
    *transformations: Tensor,
) -> Tensor:
    """Compose two transformation matrices

    Args:
        transformations: Tensors with shape
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape])

    Returns:
        Composed transformation matrix.
    """
    if len(transformations) == 0:
        raise ValueError("At least one transformation matrix must be given.")
    composition = move_channels_last(transformations[0], 2)
    for transformation in transformations[1:]:
        composition = matmul(composition, move_channels_last(transformation, 2))
    return move_channels_first(composition, 2)


def embed_matrix(matrix: Tensor, target_shape: Sequence[int]) -> Tensor:
    """Embed transformation into larger dimensional space.

    Args:
        matrix: Tensor with shape
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape])
        target_shape: Target matrix shape in form
        (n_target_output_dims + 1, n_target_input_dims + 1).

    Returns:
        Embedded matrix with shape
        ([*batch_shape, ]n_target_output_dims + 1, n_target_input_dims + 1[, *spatial_shape])
    """
    if len(target_shape) != 2:
        raise ValueError("Matrix shape must be two dimensional.")
    batch_dimensions_shape, channel_shape, spatial_shape = split_shape(
        matrix.shape, n_channel_dims=2
    )
    n_rows_needed = target_shape[0] - channel_shape[0]
    n_cols_needed = target_shape[1] - channel_shape[1]
    if n_rows_needed == 0 and n_cols_needed == 0:
        return matrix
    rows = cat(
        [
            zeros(
                n_rows_needed,
                min(channel_shape[1], channel_shape[0]),
                device=matrix.device,
                dtype=matrix.dtype,
            ),
            eye(
                n_rows_needed,
                max(0, channel_shape[1] - channel_shape[0]),
                device=matrix.device,
                dtype=matrix.dtype,
            ),
        ],
        dim=1,
    ).expand(*batch_dimensions_shape, -1, -1, *spatial_shape)
    cols = cat(
        [
            zeros(
                min(target_shape[0], channel_shape[1]),
                n_cols_needed,
                device=matrix.device,
                dtype=matrix.dtype,
            ),
            eye(
                max(0, target_shape[0] - channel_shape[1]),
                n_cols_needed,
                device=matrix.device,
                dtype=matrix.dtype,
            ),
        ],
        dim=0,
    ).expand(*batch_dimensions_shape, -1, -1, *spatial_shape)
    channel_dims = get_channel_dims(matrix.ndim, n_channel_dims=2)
    embedded_matrix = matrix
    if rows.numel() > 0:
        embedded_matrix = cat([matrix, rows], dim=channel_dims[0])
    if cols.numel() > 0:
        embedded_matrix = cat([embedded_matrix, cols], dim=channel_dims[1])
    return embedded_matrix


def convert_to_homogenous_coordinates(coordinates: Tensor, dim: int = -1) -> Tensor:
    """Converts the coordinates to homogenous coordinates.

    Args:
        coordinates: Tensor with shape
            (dim_1, ..., dim_{dim}, ..., dim_{n_dims}).

    Returns:
        Tensor with shape
        (dim_1, ..., dim_{dim + 1}, ..., dim_{n_dims}).
    """
    if dim < 0:
        dim = coordinates.ndim + dim
    homogenous_coordinates = cat(
        [
            coordinates,
            ones(1, device=coordinates.device, dtype=coordinates.dtype).expand(
                *coordinates.shape[:dim], 1, *coordinates.shape[dim + 1 :]
            ),
        ],
        dim=dim,
    )
    return homogenous_coordinates


def generate_translation_matrix(translations: Tensor) -> Tensor:
    """Generator affine translation matrix with given translations.

    Args:
        translations: Tensor with shape
            ([*batch_shape, ]n_dims[, *spatial_shape]).

    Returns:
        Translation matrix with shape (batch_size, n_dims + 1, n_dims + 1, ...).
    """
    batch_dimensions_shape, _, spatial_shape = split_shape(translations.shape, n_channel_dims=1)
    channel_dim = get_channel_dims(translations.ndim, n_channel_dims=1)[0]
    n_dims = translations.size(channel_dim)
    homogenous_translation = convert_to_homogenous_coordinates(translations, dim=-1)
    translation_matrix = cat(
        [
            cat(
                [
                    eye(n_dims, device=translations.device, dtype=translations.dtype),
                    zeros(1, n_dims, device=translations.device, dtype=translations.dtype),
                ],
                dim=0,
            ).expand(*batch_dimensions_shape, -1, -1, *spatial_shape),
            homogenous_translation.unsqueeze(channel_dim + 1),
        ],
        dim=channel_dim + 1,
    )
    return translation_matrix


def generate_scale_matrix(
    scales: Tensor,
) -> Tensor:
    """Generator scale matrix from given scales.

    Args:
        scales: Tensor with shape ([*batch_shape, ]n_dims[, *spatial_shape]).

    Returns:
        Scale matrix with shape ([*batch_shape, ]n_dims, n_dims[, *spatial_shape]).
    """
    matrix_dims = get_channel_dims(scales.ndim + 1, n_channel_dims=2)
    scales = move_channels_last(scales, n_channel_dims=1)
    return diag_embed(scales, dim1=matrix_dims[0], dim2=matrix_dims[1])


def invert_matrix(matrix: Tensor) -> Tensor:
    """Invert an affine matrix.

    Args:
        matrix: Tensor with shape
            ([*batch_shape, ]n_dims + 1, n_dims + 1[, *spatial_shape]).

    Returns:
        Inverted matrix with shape
        ([*batch_shape, ]n_dims + 1, n_dims + 1[, *spatial_shape]).
    """
    matrix = move_channels_last(matrix, 2)
    inverted_matrix = inverse(matrix)
    return move_channels_first(inverted_matrix, 2)


def add_affine_matrices(matrix_1: Tensor, matrix_2: Tensor) -> Tensor:
    """Add two affine matrices or batches of matrices

    The last row of the matrices is not included in the addition
    and is copied from the first matrix, as it should always be
    [0, ..., 0, 1] for affine transformations.

    Args:
        matrix_1: Tensor with shape
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).
        matrix_2: Tensor with shape
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).

    Returns:
        Sum matrix with shape
        ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).
    """
    matrix_1, matrix_2 = broadcast_tensors_in_parts(
        matrix_1, matrix_2, broadcast_channels=False, n_channel_dims=2
    )
    if matrix_1.shape != matrix_2.shape:
        raise ValueError("Matrices are not broadcastable.")
    matrix_1 = move_channels_last(matrix_1, 2)
    matrix_2 = move_channels_last(matrix_2, 2)
    sum_matrix = matrix_1[..., :-1, :] + matrix_2[..., :-1, :]
    sum_matrix = cat(
        [
            sum_matrix,
            matrix_1[..., -1:, :],
        ],
        dim=-2,
    )
    sum_matrix = move_channels_first(sum_matrix, 2)
    return sum_matrix


def substract_affine_matrices(matrix_1: Tensor, matrix_2: Tensor) -> Tensor:
    """Subtract two affine matrices or batches of matrices

    The last row of the matrices is not included in the subtraction
    and is copied from the first matrix, as it should always be
    [0, ..., 0, 1] for affine transformations.

    Args:
        matrix_1: Tensor with shape
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).
        matrix_2: Tensor with shape
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).

    Returns:
        Difference matrix with shape
        ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).
    """
    matrix_1, matrix_2 = broadcast_tensors_in_parts(
        matrix_1, matrix_2, broadcast_channels=False, n_channel_dims=2
    )
    if matrix_1.shape != matrix_2.shape:
        raise ValueError("Matrices are not broadcastable.")
    matrix_1 = move_channels_last(matrix_1, 2)
    matrix_2 = move_channels_last(matrix_2, 2)
    diff_matrix = matrix_1[..., :-1, :] - matrix_2[..., :-1, :]
    diff_matrix = cat(
        [
            diff_matrix,
            matrix_1[..., -1:, :],
        ],
        dim=-2,
    )
    diff_matrix = move_channels_first(diff_matrix, 2)
    return diff_matrix


def negate_affine_matrix(matrix: Tensor) -> Tensor:
    """Negate an affine matrix or a batch of matrices

    The last row of the matrix is not negated as it should always be
    [0, ..., 0, 1] for affine transformations.

    Args:
        matrix: Tensor with shape
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).

    Returns:
        Negated matrix with shape
        ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).
    """
    matrix = move_channels_last(matrix, 2)
    negated_matrix = -matrix[..., :-1, :]
    negated_matrix = cat(
        [
            negated_matrix,
            matrix[..., -1:, :],
        ],
        dim=-2,
    )
    negated_matrix = move_channels_first(negated_matrix, 2)
    return negated_matrix


def is_zero_matrix(matrix: Tensor) -> bool:
    """Is matrix a zero matrix

    Args:
        matrix: Tensor with shape
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).

    Returns:
        Whether the matrix is a zero matrix.
    """
    row_dimension = get_channel_dims(matrix.ndim, n_channel_dims=2)[0]
    return allclose(
        matrix.moveaxis(row_dimension, -1)[..., :-1],
        zeros(
            1,
            dtype=matrix.dtype,
            device=matrix.device,
        ),
        atol=ZERO_MATRIX_TOLERANCE,
    )


def is_identity_matrix(matrix: Tensor) -> bool:
    """Is a matrix an identity matrix

    Args:
        matrix: Tensor with shape
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).

    Returns:
        Whether the matrix is an identity matrix.
    """
    if matrix.size(-2) != matrix.size(-1):
        return False
    n_rows = matrix.size(get_channel_dims(matrix.ndim, n_channel_dims=2)[0])
    identity_matrix = eye(
        n_rows,
        dtype=matrix.dtype,
        device=matrix.device,
    )
    batch_shape, _, spatial_shape = split_shape(matrix.shape, n_channel_dims=2)
    broadcasted_identity_matrix = broadcast_to_in_parts(
        identity_matrix, batch_shape=batch_shape, spatial_shape=spatial_shape, n_channel_dims=2
    )
    return broadcasted_identity_matrix.shape == matrix.shape and allclose(
        matrix, broadcasted_identity_matrix, atol=IDENTITY_MATRIX_TOLERANCE
    )


def transform_values_with_affine_matrix(
    transformation_matrix: Tensor, values: Tensor, n_channel_dims: int = 1
) -> Tensor:
    """Transform values with affine matrix.

    Args:
        transformation_matrix: Tensor with shape
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).
        values: Tensor with shape
            ([*batch_shape, *channels_shape[:-1], ]n_input_dims[, *spatial_shape]).
        n_channel_dims: Number of channel dimensions in the values tensor.

    Returns:
        Transformed values with shape
        ([*batch_shape, *channels_shape[:-1], ]n_output_dims[, *spatial_shape]).
    """
    values, transformation_matrix = broadcast_tensors_in_parts(
        values, transformation_matrix, broadcast_channels=False, n_channel_dims=(n_channel_dims, 2)
    )
    transformation_matrix = move_channels_last(transformation_matrix, 2)
    if n_channel_dims > 2:
        transformation_matrix = transformation_matrix[
            (...,) + (None,) * (n_channel_dims - 2) + 2 * (slice(None),)
        ]
    values = move_channels_last(values, n_channel_dims)
    transformed = matmul(
        transformation_matrix[..., :-1, :],
        convert_to_homogenous_coordinates(values, dim=-1)[..., None],
    )[..., 0]
    transformed = move_channels_first(transformed, n_channel_dims)

    return transformed


def clear_translation_from_affine_matrix(matrix: Tensor) -> Tensor:
    """Clear translation from affine matrix

    Args:
        matrix: Tensor with shape
            ([*batch_shape, ]n_output_dims + 1, n_input_dims + 1[, *spatial_shape]).

    Returns:
        Affine matrix with cleared translation.
    """
    matrix_shape = get_channels_shape(matrix.shape, n_channel_dims=2)
    matrix = move_channels_last(matrix, 2)
    matrix = matrix[..., :-1, :-1]
    matrix = move_channels_first(matrix, 2)
    return embed_matrix(
        matrix,
        target_shape=matrix_shape,
    )

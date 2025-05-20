import numpy as np
import torch
from .perlin2d import interpolant

def generate_perlin_noise_3d_torch(
        shape, res, tileable=(False, False, False),
        interpolant=interpolant, rand1=0, rand2=0
):
    """Generate a 3D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of three ints).
            This must be a multiple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of three ints). Note shape must be a multiple
            of res.
        tileable: If the noise should be tileable along each axis
            (tuple of three bools). Defaults to (False, False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid_torch = torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1]), torch.arange(0, res[2], delta[2]))
    grid_torch = torch.cat((grid_torch[0].unsqueeze(0), grid_torch[1].unsqueeze(0), grid_torch[2].unsqueeze(0)), dim=0).cuda()
    grid_torch = grid_torch.permute(1, 2, 3, 0) % 1
    # Gradients
    theta_torch = 2 * np.pi * torch.from_numpy(rand1).cuda()
    phi_torch = 2 * np.pi * torch.from_numpy(rand2).cuda()
    gradients_torch = torch.stack(
        (torch.sin(phi_torch) * torch.cos(theta_torch), torch.sin(phi_torch) * torch.sin(theta_torch), torch.cos(phi_torch)),
        dim=3
    )

    if tileable[0]:
        gradients_torch[-1,:,:] = gradients_torch[0,:,:]
    if tileable[1]:
        gradients_torch[:, -1, :] = gradients_torch[:, 0, :]
    if tileable[2]:
        gradients_torch[:,:,-1] = gradients_torch[:,:,0]

    gradients_torch = gradients_torch.repeat_interleave(d[0], 0).repeat_interleave(d[1], 1).repeat_interleave(d[2], 2)

    g000_torch = gradients_torch[:-d[0], :-d[1], :-d[2]]
    g100_torch = gradients_torch[d[0]:, :-d[1], :-d[2]]
    g010_torch = gradients_torch[:-d[0], d[1]:, :-d[2]]
    g110_torch = gradients_torch[d[0]:, d[1]:, :-d[2]]
    g001_torch = gradients_torch[:-d[0], :-d[1], d[2]:]
    g101_torch = gradients_torch[d[0]:, :-d[1], d[2]:]
    g011_torch = gradients_torch[:-d[0], d[1]:, d[2]:]
    g111_torch = gradients_torch[d[0]:, d[1]:, d[2]:]

    # Ramps
    n000_torch = torch.sum(torch.stack((grid_torch[:, :, :, 0], grid_torch[:, :, :, 1], grid_torch[:, :, :, 2]), dim=3) * g000_torch, dim=3)
    n100_torch = torch.sum(torch.stack((grid_torch[:, :, :, 0] - 1, grid_torch[:, :, :, 1], grid_torch[:, :, :, 2]), dim=3) * g100_torch, dim=3)
    n010_torch = torch.sum(torch.stack((grid_torch[:, :, :, 0], grid_torch[:, :, :, 1] - 1, grid_torch[:, :, :, 2]), dim=3) * g010_torch, dim=3)
    n110_torch = torch.sum(torch.stack((grid_torch[:, :, :, 0] - 1, grid_torch[:, :, :, 1] - 1, grid_torch[:, :, :, 2]), dim=3) * g110_torch, dim=3)
    n001_torch = torch.sum(torch.stack((grid_torch[:, :, :, 0], grid_torch[:, :, :, 1], grid_torch[:, :, :, 2] - 1), dim=3) * g001_torch, dim=3)
    n101_torch = torch.sum(torch.stack((grid_torch[:, :, :, 0] - 1, grid_torch[:, :, :, 1], grid_torch[:, :, :, 2] - 1), dim=3) * g101_torch, dim=3)
    n011_torch = torch.sum(torch.stack((grid_torch[:, :, :, 0], grid_torch[:, :, :, 1] - 1, grid_torch[:, :, :, 2] - 1), dim=3) * g011_torch, dim=3)
    n111_torch = torch.sum(torch.stack((grid_torch[:, :, :, 0] - 1, grid_torch[:, :, :, 1] - 1, grid_torch[:, :, :, 2] - 1), dim=3) * g111_torch, dim=3)

    # Interpolation
    #grid_torch = grid_torch + (0.001 ** 0.5) * torch.randn(grid_torch.shape).cuda()

    t_torch = interpolant(grid_torch)
    n00_torch = n000_torch*(1-t_torch[:,:,:,0]) + t_torch[:,:,:,0]*n100_torch
    n10_torch = n010_torch*(1-t_torch[:,:,:,0]) + t_torch[:,:,:,0]*n110_torch
    n01_torch = n001_torch*(1-t_torch[:,:,:,0]) + t_torch[:,:,:,0]*n101_torch
    n11_torch = n011_torch*(1-t_torch[:,:,:,0]) + t_torch[:,:,:,0]*n111_torch
    n0_torch = (1-t_torch[:,:,:,1])*n00_torch + t_torch[:,:,:,1]*n10_torch
    n1_torch = (1-t_torch[:,:,:,1])*n01_torch + t_torch[:,:,:,1]*n11_torch
    img = ((1-t_torch[:,:,:,2])*n0_torch + t_torch[:,:,:,2]*n1_torch)
    return img

def generate_fractal_noise_3d_torch(
        shape, res, octaves=1, persistence=0.5, lacunarity=2,
        tileable=(False, False, False), interpolant=interpolant, rand1=None, rand2=None
):
    """Generate a 3D numpy array of fractal noise.

    Args:
        shape: The shape of the generated array (tuple of three ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of three ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of three bools). Defaults to (False, False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of fractal noise and of shape shape generated by
        combining several octaves of perlin noise.

    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """
    noise = torch.zeros(shape).cuda()
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_3d_torch(
            shape,
            (frequency*res[0], frequency*res[1], frequency*res[2]),
            tileable,
            interpolant, rand1[_], rand2[_]
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise

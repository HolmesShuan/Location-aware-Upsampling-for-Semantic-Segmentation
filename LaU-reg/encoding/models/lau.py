from __future__ import absolute_import, division

import torch
from torch.autograd import Variable

import numpy as np
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates


def th_flatten(a):
    """Flatten tensor"""
    return a.contiguous().view(a.nelement())


def th_repeat(a, repeats, axis=0):
    """Torch version of np.repeat for 1D"""
    assert len(a.size()) == 1
    return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))


def np_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.shape) == 2
    a = np.expand_dims(a, 0)
    a = np.tile(a, [repeats, 1, 1])
    return a

def np_repeat_3d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.shape) == 3
    a = np.expand_dims(a, 0)
    a = np.tile(a, [repeats, 1, 1, 1])
    return a


def th_gather_2d(input, coords):
    inds = coords[:, 0]*input.size(1) + coords[:, 1]
    x = torch.index_select(th_flatten(input), 0, inds)
    return x.view(coords.size(0))


def th_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates
    Note that coords is transposed and only 2D is supported
    Parameters
    ----------
    input : tf.Tensor. shape = (s, s)
    coords : tf.Tensor. shape = (n_points, 2)
    """

    assert order == 1
    input_size = input.size(0)

    coords = torch.clamp(coords, 0, input_size - 1)
    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[:, 0], coords_rb[:, 1]], 1)
    coords_rt = torch.stack([coords_rb[:, 0], coords_lt[:, 1]], 1)

    vals_lt = th_gather_2d(input,  coords_lt.detach())
    vals_rb = th_gather_2d(input,  coords_rb.detach())
    vals_lb = th_gather_2d(input,  coords_lb.detach())
    vals_rt = th_gather_2d(input,  coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())

    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]
    return mapped_vals

def sp_batch_map_coordinates(inputs, coords):
    """Reference implementation for batch_map_coordinates"""
    # coords = coords.clip(0, inputs.shape[1] - 1)

    assert (coords.shape[2] == 2)
    height = coords[:,:,0].clip(0, inputs.shape[1] - 1)
    width = coords[:,:,1].clip(0, inputs.shape[2] - 1)
    np.concatenate((np.expand_dims(height, axis=2), np.expand_dims(width, axis=2)), 2)

    mapped_vals = np.array([
        sp_map_coordinates(input, coord.T, mode='nearest', order=1)
        for input, coord in zip(inputs, coords)
    ])
    return mapped_vals

def th_batch_map_coordinates(input, output, coords, idx):
    """Batch version of th_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (b*c, h, w)
    output : tf.Tensor. shape = (b*c, H, W)
    coords : tf.Tensor. shape = (b*c, H*W, 2)
    Returns
    -------
    tf.Tensor. shape = (b*c, H, W)
    """

    batch_size = output.size(0) # b*c
    output_height = output.size(1) # H
    output_width = output.size(2) # W
    input_height = input.size(1) # H
    input_width = input.size(2) # W

    n_coords = coords.size(1) # h*w
    n_coords_new = output_height*output_width

    coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1), 
                torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1)), 2)

    # assert (coords.size(1) == n_coords)

    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2) # coords_lt[..., 0] : row | coords_lt[..., 1] : col
    coords_rt = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)

    lt_rb_h_ne = torch.ne(coords_lt[..., 0], coords_rb[..., 0]).float() 
    lt_rb_w_ne = torch.ne(coords_lt[..., 1], coords_rb[..., 1]).float()

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([
            idx.to(coords.device), th_flatten(coords[..., 0]), th_flatten(coords[..., 1])
        ], 1)
        inds = indices[:, 0]*input.size(1)*input.size(2) + indices[:, 1]*input.size(2) + indices[:, 2]
        vals = th_flatten(input).index_select(0, inds)
        vals = vals.view(batch_size, n_coords)
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt.detach()) # left top
    vals_rb = _get_vals_by_coords(input, coords_rb.detach()) # right bottom
    vals_lb = _get_vals_by_coords(input, coords_lb.detach()) # left bottom
    vals_rt = _get_vals_by_coords(input, coords_rt.detach()) # right top
    
    coords_offset_lt = 1.0 - torch.abs(coords_lt.detach().float() - coords)
    vals_lt = coords_offset_lt[..., 0]*coords_offset_lt[..., 1]*vals_lt
    output.copy_(vals_lt.view(batch_size, output_height, output_width))

    coords_offset_rb = 1.0 - torch.abs(coords_rb.detach().float() - coords)
    vals_rb = coords_offset_rb[..., 0]*coords_offset_rb[..., 1]*vals_rb*lt_rb_h_ne*lt_rb_w_ne
    output += vals_rb.view(batch_size, output_height, output_width)

    coords_offset_lb = 1.0 - torch.abs(coords_lb.detach().float() - coords)
    vals_lb = coords_offset_lb[..., 0]*coords_offset_lb[..., 1]*vals_lb*lt_rb_h_ne
    output += vals_lb.view(batch_size, output_height, output_width)

    coords_offset_rt = 1.0 - torch.abs(coords_rt.detach().float() - coords)
    vals_rt = coords_offset_rt[..., 0]*coords_offset_rt[..., 1]*vals_rt*lt_rb_w_ne
    output += vals_rt.view(batch_size, output_height, output_width)
    '''
    coords_offset_lt = 1 - torch.abs(coords_lt.type(coords.data.type()) - coords)
    vals_lt = coords_offset_lt[..., 0]*coords_offset_lt[..., 1]*vals_lt
    output.copy_(vals_lt.view(batch_size, output_height, output_width))

    coords_offset_rb = 1 - torch.abs(coords_rb.type(coords.data.type()) - coords)
    vals_rb = coords_offset_rb[..., 0]*coords_offset_rb[..., 1]*vals_rb*lt_rb_h_ne*lt_rb_w_ne
    output += vals_rb.view(batch_size, output_height, output_width)

    coords_offset_lb = 1 - torch.abs(coords_lb.type(coords.data.type()) - coords)
    vals_lb = coords_offset_lb[..., 0]*coords_offset_lb[..., 1]*vals_lb*lt_rb_h_ne
    output += vals_lb.view(batch_size, output_height, output_width)

    coords_offset_rt = 1 - torch.abs(coords_rt.type(coords.data.type()) - coords)
    vals_rt = coords_offset_rt[..., 0]*coords_offset_rt[..., 1]*vals_rt*lt_rb_w_ne
    output += vals_rt.view(batch_size, output_height, output_width)
    '''
    return output

def th_det_batch_map_coordinates(input, output0, output1, output2, output3, output4, coords, idx):
    """Batch version of th_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (b*c, h, w)
    output : tf.Tensor. shape = (b*c, H, W)
    coords : tf.Tensor. shape = (b*c, H*W, 2)
    Returns
    -------
    tf.Tensor. shape = (b*c, H, W)
    """

    batch_size = output0.size(0) # b*c
    output_height = output0.size(1) # H
    output_width = output0.size(2) # W
    input_height = input.size(1) # H
    input_width = input.size(2) # W

    n_coords = coords.size(1) # h*w
    n_coords_new = output_height*output_width

    coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1), 
                torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1)), 2)

    # assert (coords.size(1) == n_coords)

    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2) # coords_lt[..., 0] : row | coords_lt[..., 1] : col
    coords_rt = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)

    lt_rb_h_ne = torch.ne(coords_lt[..., 0], coords_rb[..., 0]).float() 
    lt_rb_w_ne = torch.ne(coords_lt[..., 1], coords_rb[..., 1]).float()

    def _get_vals_by_coords(input, coords):
        indices = torch.stack([
            idx.to(coords.device), th_flatten(coords[..., 0]), th_flatten(coords[..., 1])
        ], 1)
        inds = indices[:, 0]*input.size(1)*input.size(2) + indices[:, 1]*input.size(2) + indices[:, 2]
        vals = th_flatten(input).index_select(0, inds)
        vals = vals.view(batch_size, n_coords)
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt.detach()) # left top
    vals_rb = _get_vals_by_coords(input, coords_rb.detach()) # right bottom
    vals_lb = _get_vals_by_coords(input, coords_lb.detach()) # left bottom
    vals_rt = _get_vals_by_coords(input, coords_rt.detach()) # right top

    output1.copy_(vals_lt.view(batch_size, output_height, output_width))
    output2.copy_(vals_lb.view(batch_size, output_height, output_width))
    output3.copy_(vals_rt.view(batch_size, output_height, output_width))
    output4.copy_(vals_rb.view(batch_size, output_height, output_width))
    
    coords_offset_lt = 1.0 - torch.abs(coords_lt.detach().float() - coords)
    vals_lt = coords_offset_lt[..., 0]*coords_offset_lt[..., 1]*vals_lt
    output0.copy_(vals_lt.view(batch_size, output_height, output_width))

    coords_offset_rb = 1.0 - torch.abs(coords_rb.detach().float() - coords)
    vals_rb = coords_offset_rb[..., 0]*coords_offset_rb[..., 1]*vals_rb*lt_rb_h_ne*lt_rb_w_ne
    output0 += vals_rb.view(batch_size, output_height, output_width)

    coords_offset_lb = 1.0 - torch.abs(coords_lb.detach().float() - coords)
    vals_lb = coords_offset_lb[..., 0]*coords_offset_lb[..., 1]*vals_lb*lt_rb_h_ne
    output0 += vals_lb.view(batch_size, output_height, output_width)

    coords_offset_rt = 1.0 - torch.abs(coords_rt.detach().float() - coords)
    vals_rt = coords_offset_rt[..., 0]*coords_offset_rt[..., 1]*vals_rt*lt_rb_w_ne
    output0 += vals_rt.view(batch_size, output_height, output_width)
    '''
    coords_offset_lt = 1 - torch.abs(coords_lt.type(coords.data.type()) - coords)
    vals_lt = coords_offset_lt[..., 0]*coords_offset_lt[..., 1]*vals_lt
    output.copy_(vals_lt.view(batch_size, output_height, output_width))

    coords_offset_rb = 1 - torch.abs(coords_rb.type(coords.data.type()) - coords)
    vals_rb = coords_offset_rb[..., 0]*coords_offset_rb[..., 1]*vals_rb*lt_rb_h_ne*lt_rb_w_ne
    output += vals_rb.view(batch_size, output_height, output_width)

    coords_offset_lb = 1 - torch.abs(coords_lb.type(coords.data.type()) - coords)
    vals_lb = coords_offset_lb[..., 0]*coords_offset_lb[..., 1]*vals_lb*lt_rb_h_ne
    output += vals_lb.view(batch_size, output_height, output_width)

    coords_offset_rt = 1 - torch.abs(coords_rt.type(coords.data.type()) - coords)
    vals_rt = coords_offset_rt[..., 0]*coords_offset_rt[..., 1]*vals_rt*lt_rb_w_ne
    output += vals_rt.view(batch_size, output_height, output_width)
    '''
    return output0, output1, output2, output3, output4


def sp_batch_map_offsets(input, offsets):
    """Reference implementation for tf_batch_map_offsets"""

    batch_size = input.shape[0]
    input_height = input.shape[1]
    input_width = input.shape[2]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_height, :input_width], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    # coords = coords.clip(0, input_size - 1)

    mapped_vals = sp_batch_map_coordinates(input, coords)
    return mapped_vals


def th_generate_grid(batch_size, input_height, input_width, dtype, cuda):
    grid = np.meshgrid(
        range(input_height), range(input_width), indexing='ij'
    )
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)

    grid = np_repeat_2d(grid, batch_size)
    grid = torch.from_numpy(grid).type(dtype)
    if cuda:
        grid = grid.cuda()
    return Variable(grid, requires_grad=False)

def th_generate_grid_for_loss(batch_size, input_height, input_width, dtype, cuda):
    grid = np.meshgrid(
        range(input_height), range(input_width), indexing='ij'
    )
    grid = np.stack(grid, axis=0)

    grid = np_repeat_3d(grid, batch_size)
    grid = torch.from_numpy(grid).type(dtype)
    if cuda:
        grid = grid.cuda()
    return Variable(grid, requires_grad=False)

def th_generate_idx(batch_size, input_height, input_width, dtype, cuda):
    idx = th_repeat(torch.arange(0, batch_size), input_height*input_width).long()
    if cuda:
        idx = idx.cuda() # batch ID of each pixel
    return Variable(idx, requires_grad=False)

def th_batch_map_offsets(input, output, offsets, idx, grid, k=1.0, with_offset=True):
    """Batch map offsets into input
    Parameters
    ---------
    input : torch.Tensor.shape = (b*c, h, w)
    output : output.Tensor.shape = (b*c, H, W)
    offsets: torch.Tensor.shape = (b*c, H, W, 2)
    grid : torch.Tensor.shape = (b*c, H, W, 2)
    Returns
    -------
    torch.Tensor. shape = (b*c, H, W)
    """
    batch_size = input.size(0) # b*c
    input_height = input.size(1) # h
    input_width = input.size(2) # w

    if with_offset:
        offsets = offsets.view(batch_size, -1, 2) # (b*c h*w 2)
        coords = offsets + grid.to(offsets.device).div(k)
    else:
        coords = grid.to(input.device).div(k)
    mapped_vals = th_batch_map_coordinates(input, output, coords, idx)
    return mapped_vals

def th_det_batch_map_offsets(input, output0, output1, output2, output3, output4, offsets, idx, grid, k=1.0, with_offset=False):
    """Batch map offsets into input
    Parameters
    ---------
    input : torch.Tensor.shape = (b*c, h, w)
    output : output.Tensor.shape = (b*c, H, W)
    offsets: torch.Tensor.shape = (b*c, H, W, 2)
    grid : torch.Tensor.shape = (b*c, H, W, 2)
    Returns
    -------
    torch.Tensor. shape = (b*c, H, W)
    """
    batch_size = input.size(0) # b*c
    input_height = input.size(1) # h
    input_width = input.size(2) # w

    coords = grid.to(input.device).div(k)
    mapped_vals = th_det_batch_map_coordinates(input, output0, output1, output2, output3, output4, coords, idx)
    return mapped_vals
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch


def gaussian_2d(shape, sigma=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1, **kwargs):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    # these lrtb are the distance of center point to four edges
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float): Gaussian_overlap. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)



def centerpoint_radius_func1(det_size, min_overlap=0.5):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    return r1


def centerpoint_radius_func2(det_size, min_overlap=0.5):
    height, width = det_size
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    return r2

def centerpoint_radius_func3(det_size, min_overlap=0.5):
    height, width = det_size
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return r3


def maxwh_radius_func(det_size, min_overlap=None, scale=1):
    height, width = det_size
    return max(det_size) * scale



def xy_wh_r_2_xy_sigma(xywhr, sigma_factor=1):
    """Convert oriented bounding box to 2-D Gaussian distribution.
    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).
    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    assert False
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh) * sigma_factor

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def gaussian_2d_2(shape, sigma):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (numpy.ndarray): gauss covariance matrix. shape 2x2.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    assert False
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    new_y = np.repeat(y, x.shape[1], axis=1)
    new_x = np.repeat(x, y.shape[0], axis=0)
    coords = np.stack([new_y, new_x], axis=-1).reshape(-1, 2)
    h = np.matmul(np.matmul(coords, np.linalg.inv(sigma)), coords.T)
    h = np.diagonal(h)
    h = np.exp(-h / 2)
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian_2(heatmap, center, radius, sigma, extend_radius=None, extend_ratio=None, **kwargs):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (torch.Tensor): Radius of gausian. shape 2
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    assert False
    x_radius, y_radius = int(radius[0]), int(radius[1])
    if extend_radius is not None:
        assert extend_ratio is None
        extend_x_radius = x_radius + extend_radius
        extend_y_radius = y_radius + extend_radius
    elif extend_ratio is not None:
        extend_x_radius = x_radius * extend_ratio
        extend_y_radius = y_radius * extend_ratio
    else:
        raise NotImplementedError

    gaussian = gaussian_2d_2((2 * extend_x_radius + 1, 2 * extend_y_radius + 1), sigma=sigma)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    # these lrtb are the distance of center point to four edges
    left, right = min(x, extend_x_radius), min(width - x, extend_x_radius + 1)
    top, bottom = min(y, extend_y_radius), min(height - y, extend_y_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[extend_y_radius - top:extend_y_radius + bottom,
                 extend_x_radius - left:extend_x_radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
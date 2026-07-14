import torch
from einops.einops import rearrange

def rgb_to_ycbcr(image, device):
    rgb_array = rearrange(image, '1 c h w ->1 h w c')

    transform_matrix = torch.tensor([[0.299, 0.587, 0.114],
                                     [-0.169, -0.331, 0.5],
                                     [0.5, -0.419, -0.081]]).to(device)

    ycbcr_array = torch.matmul(rgb_array, transform_matrix.T)

    y_channel = ycbcr_array[:, :, :, 0]
    cb_channel = ycbcr_array[:, :, :, 1]
    cr_channel = ycbcr_array[:, :, :, 2]

    y_channel = y_channel.clamp(0.0, 255.0)
    return y_channel.unsqueeze(0), cb_channel.unsqueeze(0), cr_channel.unsqueeze(0)


def ycbcr_to_rgb(y, cb, cr, device):
    ycbcr_array = torch.cat([y, cb, cr], dim=1)
    ycbcr_array = rearrange(ycbcr_array, '1 c h w ->1 h w c')

    transform_matrix = torch.tensor([[1, 0, 1.402],
                                     [1, -0.344136, -0.714136],
                                     [1, 1.772, 0]]).to(device)

    rgb_array = torch.matmul(ycbcr_array, transform_matrix.T)

    rgb_array = torch.clamp(rgb_array, 0, 1.0)
    rgb_array = rearrange(rgb_array, 'b h w c-> b c h w')

    return rgb_array
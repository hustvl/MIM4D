import torch
from .smooth_sampler import SmoothSampler
from .voxel_pool import voxel_pool
from .point_ops import group_inner_inds

__all__ = ['SmoothSampler', 'voxel_pool', 'group_inner_inds']


# def grid_sample_3d(feature, grid):
#     N, C, ID, IH, IW = feature.shape
#     _, D, H, W, _ = grid.shape
# 
#     ix = grid[..., 0]
#     iy = grid[..., 1]
#     iz = grid[..., 2]
# 
#     ix = ((ix + 1) / 2) * (IW - 1)
#     iy = ((iy + 1) / 2) * (IH - 1)
#     iz = ((iz + 1) / 2) * (ID - 1)
#     with torch.no_grad():
#         
#         ix_tnw = torch.floor(ix)
#         iy_tnw = torch.floor(iy)
#         iz_tnw = torch.floor(iz)
# 
#         ix_tne = ix_tnw + 1
#         iy_tne = iy_tnw
#         iz_tne = iz_tnw
# 
#         ix_tsw = ix_tnw
#         iy_tsw = iy_tnw + 1
#         iz_tsw = iz_tnw
# 
#         ix_tse = ix_tnw + 1
#         iy_tse = iy_tnw + 1
#         iz_tse = iz_tnw
# 
#         ix_bnw = ix_tnw
#         iy_bnw = iy_tnw
#         iz_bnw = iz_tnw + 1
# 
#         ix_bne = ix_tnw + 1
#         iy_bne = iy_tnw
#         iz_bne = iz_tnw + 1
# 
#         ix_bsw = ix_tnw
#         iy_bsw = iy_tnw + 1
#         iz_bsw = iz_tnw + 1
# 
#         ix_bse = ix_tnw + 1
#         iy_bse = iy_tnw + 1
#         iz_bse = iz_tnw + 1
# 
#     tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
#     tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
#     tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
#     tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
#     bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
#     bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
#     bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
#     bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)
# 
# 
#     with torch.no_grad():
# 
#         torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
#         torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
#         torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)
# 
#         torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
#         torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
#         torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)
# 
#         torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
#         torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
#         torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)
# 
#         torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
#         torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
#         torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)
# 
#         torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
#         torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
#         torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)
# 
#         torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
#         torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
#         torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)
# 
#         torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
#         torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
#         torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)
# 
#         torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
#         torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
#         torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)
# 
#     feature = feature.view(N, C, ID * IH * IW)
# 
#     tnw_val = torch.gather(feature, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     tne_val = torch.gather(feature, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     # tsw_val = torch.gather(feature, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     # tse_val = torch.gather(feature, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     # bnw_val = torch.gather(feature, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     bne_val = torch.gather(feature, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     bsw_val = torch.gather(feature, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
#     bse_val = torch.gather(feature, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))
# 
# 
#     out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
#                tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
#             #    tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
#             #    tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
#             #    bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
#                bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
#                bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
#                bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W)
#                )
# 
#     return out_val

def grid_sample_3d(volume, optical):
    """
    bilinear sampling cannot guarantee continuous first-order gradient
    mimic pytorch grid_sample function
    The 8 corner points of a volume noted as: 4 points (front view); 4 points (back view)
    fnw (front north west) point
    bse (back south east) point
    :param volume: [B, C, X, Y, Z]
    :param optical: [B, x, y, z, 3]
    :return:
    """
    N, C, ID, IH, IW = volume.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]


    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)

    mask_x = (ix > 0) & (ix < IW)
    mask_y = (iy > 0) & (iy < IH)
    mask_z = (iz > 0) & (iz < ID)

    mask = mask_x & mask_y & mask_z  # [B, x, y, z]
    mask = mask[:, None, :, :, :].repeat(1, C, 1, 1, 1)  # [B, C, x, y, z]

    with torch.no_grad():
        # back north west
        ix_bnw = torch.floor(ix)
        iy_bnw = torch.floor(iy)
        iz_bnw = torch.floor(iz)

        ix_bne = ix_bnw + 1
        iy_bne = iy_bnw
        iz_bne = iz_bnw

        ix_bsw = ix_bnw
        iy_bsw = iy_bnw + 1
        iz_bsw = iz_bnw

        ix_bse = ix_bnw + 1
        iy_bse = iy_bnw + 1
        iz_bse = iz_bnw

        # front view
        ix_fnw = ix_bnw
        iy_fnw = iy_bnw
        iz_fnw = iz_bnw + 1

        ix_fne = ix_bnw + 1
        iy_fne = iy_bnw
        iz_fne = iz_bnw + 1

        ix_fsw = ix_bnw
        iy_fsw = iy_bnw + 1
        iz_fsw = iz_bnw + 1

        ix_fse = ix_bnw + 1
        iy_fse = iy_bnw + 1
        iz_fse = iz_bnw + 1

    # back view
    bnw = (ix_fse - ix) * (iy_fse - iy) * (iz_fse - iz)  # smaller volume, larger weight
    bne = (ix - ix_fsw) * (iy_fsw - iy) * (iz_fsw - iz)
    bsw = (ix_fne - ix) * (iy - iy_fne) * (iz_fne - iz)
    bse = (ix - ix_fnw) * (iy - iy_fnw) * (iz_fnw - iz)

    # front view
    fnw = (ix_bse - ix) * (iy_bse - iy) * (iz - iz_bse)  # smaller volume, larger weight
    fne = (ix - ix_bsw) * (iy_bsw - iy) * (iz - iz_bsw)
    fsw = (ix_bne - ix) * (iy - iy_bne) * (iz - iz_bne)
    fse = (ix - ix_bnw) * (iy - iy_bnw) * (iz - iz_bnw)

    with torch.no_grad():
        # back view
        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

        # front view
        torch.clamp(ix_fnw, 0, IW - 1, out=ix_fnw)
        torch.clamp(iy_fnw, 0, IH - 1, out=iy_fnw)
        torch.clamp(iz_fnw, 0, ID - 1, out=iz_fnw)

        torch.clamp(ix_fne, 0, IW - 1, out=ix_fne)
        torch.clamp(iy_fne, 0, IH - 1, out=iy_fne)
        torch.clamp(iz_fne, 0, ID - 1, out=iz_fne)

        torch.clamp(ix_fsw, 0, IW - 1, out=ix_fsw)
        torch.clamp(iy_fsw, 0, IH - 1, out=iy_fsw)
        torch.clamp(iz_fsw, 0, ID - 1, out=iz_fsw)

        torch.clamp(ix_fse, 0, IW - 1, out=ix_fse)
        torch.clamp(iy_fse, 0, IH - 1, out=iy_fse)
        torch.clamp(iz_fse, 0, ID - 1, out=iz_fse)

    # xxx = volume[:, :, iz_bnw.long(), iy_bnw.long(), ix_bnw.long()]
    volume = volume.view(N, C, ID * IH * IW)
    # yyy = volume[:, :, (iz_bnw * ID + iy_bnw * IW + ix_bnw).long()]

    # back view
    bnw_val = torch.gather(volume, 2,
                           (iz_bnw * ID ** 2 + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(volume, 2,
                           (iz_bne * ID ** 2 + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(volume, 2,
                           (iz_bsw * ID ** 2 + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(volume, 2,
                           (iz_bse * ID ** 2 + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    # front view
    fnw_val = torch.gather(volume, 2,
                           (iz_fnw * ID ** 2 + iy_fnw * IW + ix_fnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    fne_val = torch.gather(volume, 2,
                           (iz_fne * ID ** 2 + iy_fne * IW + ix_fne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    fsw_val = torch.gather(volume, 2,
                           (iz_fsw * ID ** 2 + iy_fsw * IW + ix_fsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    fse_val = torch.gather(volume, 2,
                           (iz_fse * ID ** 2 + iy_fse * IW + ix_fse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (
        # back
            bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
            bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
            bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
            bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W) +
            # front
            fnw_val.view(N, C, D, H, W) * fnw.view(N, 1, D, H, W) +
            fne_val.view(N, C, D, H, W) * fne.view(N, 1, D, H, W) +
            fsw_val.view(N, C, D, H, W) * fsw.view(N, 1, D, H, W) +
            fse_val.view(N, C, D, H, W) * fse.view(N, 1, D, H, W)

    )

    # * zero padding
    out_val = torch.where(mask, out_val, torch.zeros_like(out_val).float().to(out_val.device))

    return out_val
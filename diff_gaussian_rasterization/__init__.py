#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    scores,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        scores,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        scores,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        # Split the SH coefficients into the direct color (dc, the first coefficient)
        # and the remaining SH coefficients expected by the CUDA implementation.
        if sh.numel() != 0:
            # dc : [P, 3]
            dc = sh[:, 0, :].contiguous()
            sh_rest = sh[:, 1:, :].contiguous()
        else:
            # Empty tensors keep interface identical when SHs are not provided
            dc = torch.Tensor([])
            sh_rest = torch.Tensor([])

        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            dc,
            sh_rest,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug,
            raster_settings.tile_size,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, num_buckets, color, invdepth, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, num_buckets, color, invdepth, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = _C.rasterize_gaussians(*args)

        # Save values required for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered  # R
        ctx.num_buckets = num_buckets    # B

        ctx.save_for_backward(
            colors_precomp,
            opacities,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            dc,
            sh_rest,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            sampleBuffer,
            invdepth,
        )

        # Keep the original return signature (color, radii, depth) expected by calling code.
        return color, radii, invdepth

    @staticmethod
    def backward(ctx, grad_out_color, _0, grad_out_invdepth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (colors_precomp,
         opacities,
         means3D,
         scales,
         rotations,
         cov3Ds_precomp,
         radii,
         dc,
         sh_rest,
         geomBuffer,
         binningBuffer,
         imgBuffer,
         sampleBuffer,
         invdepth_saved) = ctx.saved_tensors

        # Handle missing gradient for invdepth (can be None if not used)
        if grad_out_invdepth is None:
            grad_out_invdepth = torch.Tensor([])

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            dc,
            sh_rest,
            grad_out_invdepth,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            ctx.num_rendered,
            binningBuffer,
            imgBuffer,
            ctx.num_buckets,
            sampleBuffer,
            raster_settings.antialiasing,
            raster_settings.debug,
            raster_settings.tile_size,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_dc, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        # Combine gradients of dc (first SH coefficient) and the remaining SH coefficients
        if grad_sh.numel() != 0 or grad_dc.numel() != 0:
            grad_sh_full = torch.cat([grad_dc, grad_sh], dim=1)
        else:
            grad_sh_full = torch.Tensor([])

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh_full,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,  # scores (not used)
            None,  # raster_settings
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool = False  # 新增抗锯齿开关，默认关闭
    tile_size : int = 16  # 添加tile_size字段，默认值16

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, scores = None, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scores is None:
            scores = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            scores,
            raster_settings,
        )


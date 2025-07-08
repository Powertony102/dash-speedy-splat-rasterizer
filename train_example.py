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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def get_optimal_tile_size(image_height, image_width):
    """根据图像分辨率动态选择tile_size"""
    if image_height <= 256 and image_width <= 256:
        return 32  # 低分辨率用较大tile
    elif image_height <= 512 and image_width <= 512:
        return 24  # 中等分辨率
    else:
        return 16  # 高分辨率用较小tile

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, render_size=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 计算渲染尺寸
    image_height = int(viewpoint_camera.image_height) if render_size is None else render_size[0]
    image_width = int(viewpoint_camera.image_width) if render_size is None else render_size[1]

    # 根据分辨率动态选择tile_size
    tile_size = get_optimal_tile_size(image_height, image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
        tile_size=tile_size,  # 将tile_size作为raster_settings的一部分
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)  # 不再需要传递tile_size参数
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)  # 不再需要传递tile_size参数
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out

# 使用示例
def example_usage():
    """展示如何使用新的接口"""
    
    # 方法1：使用默认tile_size (16)
    raster_settings = GaussianRasterizationSettings(
        image_height=512,
        image_width=512,
        tanfovx=0.5,
        tanfovy=0.5,
        bg=torch.tensor([0.1, 0.1, 0.1], device="cuda"),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=torch.eye(4, device="cuda"),
        sh_degree=3,
        campos=torch.tensor([0.0, 0.0, 2.0], device="cuda"),
        prefiltered=False,
        debug=False
        # tile_size默认为16
    )
    
    # 方法2：显式指定tile_size
    raster_settings_optimized = GaussianRasterizationSettings(
        image_height=256,
        image_width=256,
        tanfovx=0.5,
        tanfovy=0.5,
        bg=torch.tensor([0.1, 0.1, 0.1], device="cuda"),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=torch.eye(4, device="cuda"),
        sh_degree=3,
        campos=torch.tensor([0.0, 0.0, 2.0], device="cuda"),
        prefiltered=False,
        debug=False,
        tile_size=32  # 低分辨率使用较大tile_size
    )
    
    # 方法3：动态计算tile_size
    image_height, image_width = 256, 256
    optimal_tile_size = get_optimal_tile_size(image_height, image_width)
    
    raster_settings_dynamic = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=0.5,
        tanfovy=0.5,
        bg=torch.tensor([0.1, 0.1, 0.1], device="cuda"),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=torch.eye(4, device="cuda"),
        sh_degree=3,
        campos=torch.tensor([0.0, 0.0, 2.0], device="cuda"),
        prefiltered=False,
        debug=False,
        tile_size=optimal_tile_size  # 动态计算的tile_size
    )
    
    print(f"默认tile_size: {raster_settings.tile_size}")
    print(f"优化tile_size: {raster_settings_optimized.tile_size}")
    print(f"动态tile_size: {raster_settings_dynamic.tile_size}")

if __name__ == "__main__":
    example_usage() 
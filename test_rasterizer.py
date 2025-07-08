import torch
import numpy as np
from diff_gaussian_rasterization import rasterize_gaussians, GaussianRasterizationSettings

def main():
    print("开始测试 rasterizer...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    N = 100  # 高斯点的数量
    
    # 3D位置 (N, 3)
    means3D = torch.randn(N, 3, dtype=torch.float32, device=device) * 0.1
    
    # 2D投影位置 (N, 2) - 这里我们先用随机值，实际应该通过投影计算
    means2D = torch.randn(N, 2, dtype=torch.float32, device=device)
    
    # 球谐系数 (N, 16) - 对于degree=3的球谐
    sh = torch.randn(N, 16, dtype=torch.float32, device=device) * 0.1
    
    # 预计算颜色 (N, 3) - 空张量，因为我们使用球谐
    colors_precomp = torch.empty(0, 3, dtype=torch.float32, device=device)
    
    # 不透明度 (N, 1)
    opacities = torch.sigmoid(torch.randn(N, 1, dtype=torch.float32, device=device))
    
    # 缩放 (N, 3)
    scales = torch.randn(N, 3, dtype=torch.float32, device=device) * 0.1
    
    # 旋转 (N, 4) - 四元数
    rotations = torch.randn(N, 4, dtype=torch.float32, device=device)
    rotations = rotations / torch.norm(rotations, dim=1, keepdim=True)  # 归一化
    
    # 预计算3D协方差 - 空张量，因为我们使用scale/rotation
    cov3Ds_precomp = torch.empty(0, 6, dtype=torch.float32, device=device)
    
    # 分数 (N, 1)
    scores = torch.randn(N, 1, dtype=torch.float32, device=device)
    
    # 测试不同的tile_size
    tile_sizes = [16, 32, 64]
    
    for tile_size in tile_sizes:
        print(f"\n测试 tile_size = {tile_size}")
        try:
            # 创建光栅化设置
            image_height = 256
            image_width = 256
            tanfovx = 0.5
            tanfovy = 0.5
            bg = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device)
            scale_modifier = 1.0
            
            # 简单的视图矩阵 (4x4)
            viewmatrix = torch.eye(4, dtype=torch.float32, device=device)
            viewmatrix[2, 3] = -2.0
            
            # 简单的投影矩阵 (4x4)
            projmatrix = torch.eye(4, dtype=torch.float32, device=device)
            projmatrix[0, 0] = 1.0 / tanfovx
            projmatrix[1, 1] = 1.0 / tanfovy
            projmatrix[2, 2] = -(1000.0 + 0.1) / (1000.0 - 0.1)
            projmatrix[2, 3] = -2.0 * 1000.0 * 0.1 / (1000.0 - 0.1)
            projmatrix[3, 2] = -1.0
            projmatrix[3, 3] = 0.0
            
            sh_degree = 3
            campos = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32, device=device)
            prefiltered = False
            debug = True  # 开启调试模式
            
            raster_settings = GaussianRasterizationSettings(
                image_height=image_height,
                image_width=image_width,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg,
                scale_modifier=scale_modifier,
                viewmatrix=viewmatrix,
                projmatrix=projmatrix,
                sh_degree=sh_degree,
                campos=campos,
                prefiltered=prefiltered,
                debug=debug,
                tile_size=tile_size  # 将tile_size作为raster_settings的一部分
            )
            
            # 调用光栅化函数
            color, radii, kernel_times = rasterize_gaussians(
                means3D=means3D,
                means2D=means2D,
                sh=sh,
                colors_precomp=colors_precomp,
                opacities=opacities,
                scales=scales,
                rotations=rotations,
                cov3Ds_precomp=cov3Ds_precomp,
                scores=scores,
                raster_settings=raster_settings
            )
            
            print(f"成功! 输出形状: {color.shape}")
            print(f"颜色范围: [{color.min().item():.3f}, {color.max().item():.3f}]")
            print(f"半径形状: {radii.shape}")
            print(f"内核时间: {kernel_times}")
            
            # 同步CUDA
            torch.cuda.synchronize()
            print(f"CUDA同步成功!")
            
        except Exception as e:
            print(f"失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()
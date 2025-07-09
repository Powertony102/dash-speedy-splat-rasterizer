/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "cuda_rasterizer/rasterizer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA, "Rasterize Gaussians");
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA, "Backward pass for rasterization of gaussians");
  m.def("mark_visible", [](const torch::Tensor& means3D, const torch::Tensor& viewmatrix, const torch::Tensor& projmatrix) {
    int P = means3D.size(0);
    auto options = torch::TensorOptions().dtype(torch::kBool).device(means3D.device());
    torch::Tensor visible = torch::empty({P}, options);

    if (P > 0)
    {
        CudaRasterizer::Rasterizer::markVisible(
            P,
            (float*)means3D.contiguous().data_ptr<float>(),
            (float*)viewmatrix.contiguous().data_ptr<float>(),
            (float*)projmatrix.contiguous().data_ptr<float>(),
            visible.contiguous().data_ptr<bool>()
        );
    }
    return visible;
  }, "Marks gaussians visible now");
}
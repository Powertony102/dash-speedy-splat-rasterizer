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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA,
		"antialiasing"_a,
		"debug"_a=false,
		"tile_size"_a=16)
	.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA, "Backward pass for rasterization of gaussians")
	.def("mark_visible", &mark_visible, "Marks gaussians visible now");
}
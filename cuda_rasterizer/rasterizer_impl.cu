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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	float4* conic_opacity,
	uint32_t* tiles_touched,
	dim3 grid,
	const int tile_size)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Stop if this Gaussian is invisible
	if (tiles_touched[idx] == 0)
		return;

	// ---- OctagonSplat: Start of new duplication logic ----

	// 1. Re-calculate octagon vertices
	float4 con_o = conic_opacity[idx];
	float t = 2.f * log(255.f * con_o.w);
	// No need to check t < 0, as tiles_touched > 0 check already handled it.

	float a = con_o.x;
	float b = con_o.y;
	float c = con_o.z;

	float2 V[8];
	float b2 = b * b;
	
	float d1 = a + 2*b + c;
	if (d1 <= 0.f) return;
	float x1 = sqrtf(t * (b+c)*(b+c) / (d1 * (a*c - b2)));
	V[4].x = x1; V[4].y = -(a+b)/(b+c) * x1;
	V[5].x = -x1; V[5].y = -V[4].y;

	float d2 = a - 2*b + c;
	if (d2 <= 0.f) return;
	float x2 = sqrtf(t * (c-b)*(c-b) / (d2 * (a*c - b2)));
	V[0].x = x2; V[0].y = (a-b)/(c-b) * x2; // Fixed sign error here
	V[1].x = -x2; V[1].y = -V[0].y;
	
	if(c == 0.f) return;
	float x_m_inf = sqrtf(t * c / (a*c - b2));
	V[2].x = x_m_inf; V[2].y = -b/c * x_m_inf;
	V[3].x = -x_m_inf; V[3].y = -V[2].y;
	
	if(a == 0.f) return;
	float y_m_0 = sqrtf(t * a / (a*c - b2));
	V[6].y = y_m_0; V[6].x = -b/a * y_m_0;
	V[7].y = -y_m_0; V[7].x = -V[6].x;

	// Add center offset
	#pragma unroll
	for(int i = 0; i < 8; ++i) {
		V[i].x += points_xy[idx].x;
		V[i].y += points_xy[idx].y;
	}

	// 2. Coarse Culling: AABB
	float2 min_coord = V[0], max_coord = V[0];
	#pragma unroll
	for(int i = 1; i < 8; ++i) {
		min_coord.x = min(min_coord.x, V[i].x);
		min_coord.y = min(min_coord.y, V[i].y);
		max_coord.x = max(max_coord.x, V[i].x);
		max_coord.y = max(max_coord.y, V[i].y);
	}

	int min_tile_x = max(0, (int)floor(min_coord.x / tile_size));
	int max_tile_x = min((int)grid.x - 1, (int)floor(max_coord.x / tile_size));
	int min_tile_y = max(0, (int)floor(min_coord.y / tile_size));
	int max_tile_y = min((int)grid.y - 1, (int)floor(max_coord.y / tile_size));

	// 3. Fine Culling & Key Generation
	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	uint32_t written_count = 0;

	for (int ty = min_tile_y; ty <= max_tile_y; ++ty) {
		for (int tx = min_tile_x; tx <= max_tile_x; ++tx) {
			float2 tile_corners[4] = {
				{(float)tx * tile_size, (float)ty * tile_size},
				{(float)(tx+1) * tile_size, (float)ty * tile_size},
				{(float)tx * tile_size, (float)(ty+1) * tile_size},
				{(float)(tx+1) * tile_size, (float)(ty+1) * tile_size}
			};

			bool tile_is_outside = false;
			#pragma unroll
			for (int i = 0; i < 8; ++i) {
				float2 v1 = V[i];
				float2 v2 = V[(i + 1) % 8];
				int out_count = 0;
				#pragma unroll
				for (int j = 0; j < 4; ++j) {
					float edge_val = (tile_corners[j].x - v1.x) * (v2.y - v1.y) - (tile_corners[j].y - v1.y) * (v2.x - v1.x);
					if (edge_val > 0) out_count++;
				}
				if (out_count == 4) {
					tile_is_outside = true;
					break;
				}
			}

			if (!tile_is_outside) {
				uint32_t tile_id = ty * grid.x + tx;
				uint32_t depth_key = __float_as_uint(depths[idx]);
				uint64_t key = ((uint64_t)tile_id << 32) | ((uint64_t)depth_key);
				gaussian_keys_unsorted[off + written_count] = key;
				gaussian_values_unsorted[off + written_count] = idx;
				written_count++;
			}
		}
	}
	// ---- OctagonSplat: End of new duplication logic ----
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
  float* kernel_times,
	float* out_color,
	int* radii,
	bool debug,
	const int tile_size)
{
  // Timers for functions
  cudaEvent_t overallStart, overallStop;
  cudaEventCreate(&overallStart);
  cudaEventCreate(&overallStop);
  float milliseconds;

	int num_rendered;
  // Record Overall forward time
  cudaEventRecord(overallStart, 0);

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// 使用动态tile_size而不是固定的BLOCK_X和BLOCK_Y
	dim3 tile_grid((width + tile_size - 1) / tile_size, (height + tile_size - 1) / tile_size, 1);
	dim3 block(tile_size, tile_size, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		tile_size
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		geomState.conic_opacity,
    geomState.tiles_touched,
		tile_grid,
		tile_size)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		tile_size), debug)

  // End Overall timer
  cudaEventRecord(overallStop, 0);
  cudaEventSynchronize(overallStop);
  cudaEventElapsedTime(&milliseconds, overallStart, overallStop);
  kernel_times[0] = milliseconds;

  cudaEventDestroy(overallStart);
  cudaEventDestroy(overallStop);

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dG2,
	bool debug,
	const int tile_size)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// 使用动态tile_size而不是固定的BLOCK_X和BLOCK_Y
	const dim3 tile_grid((width + tile_size - 1) / tile_size, (height + tile_size - 1) / tile_size, 1);
	const dim3 block(tile_size, tile_size, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
    dL_dG2), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}

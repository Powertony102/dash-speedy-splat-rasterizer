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
	float4* con_o,
  uint32_t* tiles_touched,
	dim3 grid,
	const int tile_size)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (tiles_touched[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    // Update unsorted arrays with Gaussian idx for every tile that
    // Gaussian touches
    duplicateToTilesTouched(
        points_xy[idx], con_o[idx], grid,
        idx, off, depths[idx],
        gaussian_keys_unsorted,
        gaussian_values_unsorted,
        tile_size);
	}
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

// 新增：统计每个 tile 所需桶数的 kernel
__global__ void perTileBucketCount(int T, uint2* ranges, uint32_t* bucketCount)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= T)  return;

    uint2 range = ranges[idx];
    int num_splats  = range.y - range.x;

    // 需要的桶数：每 32 个 splat 加 1，但在处理第 0 个 splat 前
    // 就要保存一次采样状态，因此加上 “起始桶”。
    int num_buckets = (num_splats == 0) ? 0 : (num_splats - 1) / 32 + 1; // 修正：每 32 个 splat 归为一个 bucket，且第 0 个 splat 前也要存一次
    bucketCount[idx] = (uint32_t)num_buckets;
}

// 替换 ImageState::fromChunk，实现对 bucket 相关缓存的申请
CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
    ImageState img;
    // 逐条 obtain，保持与 header 字段顺序一致
    obtain(chunk, img.accum_alpha, N, 128);
    obtain(chunk, img.n_contrib, N, 128);
    obtain(chunk, img.ranges, N, 128);

    // bucket 统计相关
    int* dummy = nullptr;
    int* wummy = nullptr;
    cub::DeviceScan::InclusiveSum(nullptr, img.scan_size, dummy, wummy, N); // 仅为了获取 scan_size
    obtain(chunk, img.contrib_scan, img.scan_size, 128);

    obtain(chunk, img.max_contrib, N, 128);
    obtain(chunk, img.pixel_colors, N * NUM_CHANNELS_3DGS, 128);
    obtain(chunk, img.pixel_invDepths, N, 128);

    obtain(chunk, img.bucket_count, N, 128);
    obtain(chunk, img.bucket_offsets, N, 128);
    cub::DeviceScan::InclusiveSum(nullptr, img.bucket_count_scan_size, img.bucket_count, img.bucket_count, N);
    obtain(chunk, img.bucket_count_scanning_space, img.bucket_count_scan_size, 128);

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

// 更新 SampleState::fromChunk，使其支持可变 block_size（tile_size^2）
CudaRasterizer::SampleState CudaRasterizer::SampleState::fromChunk(char*& chunk, size_t C, size_t block_size)
{
    SampleState sample;
    // per-bucket 长度 = block_size
    obtain(chunk, sample.bucket_to_tile, C * block_size, 128);
    obtain(chunk, sample.T, C * block_size, 128);
    obtain(chunk, sample.ar, NUM_CHANNELS_3DGS * C * block_size, 128);
    obtain(chunk, sample.ard, C * block_size, 128);
    return sample;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
std::tuple<int,int> CudaRasterizer::Rasterizer::forward(
    std::function<char* (size_t)> geometryBuffer,
    std::function<char* (size_t)> binningBuffer,
    std::function<char* (size_t)> imageBuffer,
    std::function<char* (size_t)> sampleBuffer,
    const int P, int D, int M,
    const float* background,
    const int width, int height,
    const float* means3D,
    const float* dc,
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
    float* out_color,
    float* invdepth,
    bool antialiasing,
    int* radii,
    bool debug,
    int tile_size)
{
    int num_rendered;

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    // ---------------- Geometry buffers ------------------
    size_t geom_chunk_size = required<GeometryState>(P);
    char* geom_chunkptr = geometryBuffer(geom_chunk_size);
    GeometryState geomState = GeometryState::fromChunk(geom_chunkptr, P);

    if (radii == nullptr)
        radii = geomState.internal_radii;

    // ---------------- Tile grid -------------------------
    dim3 tile_grid((width + tile_size - 1) / tile_size, (height + tile_size - 1) / tile_size, 1);
    dim3 block(tile_size, tile_size, 1);

    // ---------------- Image buffers ---------------------
    size_t img_chunk_size = required<ImageState>(width * height);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

    if (NUM_CHANNELS_3DGS != 3 && colors_precomp == nullptr)
        throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");

    // ---------------- Preprocess ------------------------
    CHECK_CUDA(FORWARD::preprocess(
        P, D, M,
        means3D,
        (glm::vec3*)scales,
        scale_modifier,
        (glm::vec4*)rotations,
        opacities,
        dc,
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
        antialiasing,
        tile_size), debug)

    // ------------- Prefix sum over tiles_touched --------
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    // ---------------- Binning buffers -------------------
    size_t bin_chunk_size = required<BinningState>(num_rendered);
    char* bin_chunkptr = binningBuffer(bin_chunk_size);
    BinningState binningState = BinningState::fromChunk(bin_chunkptr, num_rendered);

    // Generate key-value pairs for (Gaussian, tile)
    duplicateWithKeys<<<(P + 255) / 256, 256>>>(
        P,
        geomState.means2D,
        geomState.depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        geomState.conic_opacity,
        geomState.tiles_touched,
        tile_grid,
        tile_size);

    int bit = getHigherMsb(tile_grid.x * tile_grid.y);

    // Radix sort keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space,
        binningState.sorting_size,
        binningState.point_list_keys_unsorted, binningState.point_list_keys,
        binningState.point_list_unsorted, binningState.point_list,
        num_rendered, 0, 32 + bit), debug)

    CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

    // Identify tile ranges
    if (num_rendered > 0)
        identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            binningState.point_list_keys,
            imgState.ranges);

    // ------------- Bucket preparation ------------------
    int num_tiles = tile_grid.x * tile_grid.y;
    perTileBucketCount<<<(num_tiles + 255) / 256, 256>>>(num_tiles, imgState.ranges, imgState.bucket_count);

    CHECK_CUDA(cub::DeviceScan::InclusiveSum(imgState.bucket_count_scanning_space, imgState.bucket_count_scan_size, imgState.bucket_count, imgState.bucket_offsets, num_tiles), debug)

    unsigned int bucket_sum;
    CHECK_CUDA(cudaMemcpy(&bucket_sum, imgState.bucket_offsets + num_tiles - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost), debug);

    // ------------- Sample buffers ----------------------
    // 每 bucket 需要存储：
    //   bucket_to_tile  : uint32_t  (4B)
    //   T              : float     (4B)
    //   ar             : float*NUM_CHANNELS_3DGS (4B*NUM_CHANNELS_3DGS)
    //   ard            : float     (4B)
    const size_t block_size = tile_size * tile_size;
    const size_t bytes_per_sample = sizeof(uint32_t) + sizeof(float) * (NUM_CHANNELS_3DGS + 2);
    size_t sample_chunk_size = bucket_sum * block_size * bytes_per_sample + 128; // 加 128 作为对齐余量
    char* sample_chunkptr = sampleBuffer(sample_chunk_size);
    SampleState sampleState = SampleState::fromChunk(sample_chunkptr, bucket_sum, tile_size * tile_size);

    // ------------- Rendering ---------------------------
    const float* feature_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
    CHECK_CUDA(FORWARD::render(
        tile_grid, block,
        imgState.ranges,
        binningState.point_list,
        imgState.bucket_offsets,
        sampleState.bucket_to_tile,
        sampleState.T,
        sampleState.ar,
        sampleState.ard,
        width, height,
        geomState.means2D,
        feature_ptr,
        geomState.conic_opacity,
        imgState.accum_alpha,
        imgState.n_contrib,
        imgState.max_contrib,
        background,
        out_color,
        geomState.depths,
        invdepth), debug)

    // 保存结果到 ImageState 方便 backward 使用
    CHECK_CUDA(cudaMemcpy(imgState.pixel_colors, out_color, sizeof(float) * width * height * NUM_CHANNELS_3DGS, cudaMemcpyDeviceToDevice), debug);
    CHECK_CUDA(cudaMemcpy(imgState.pixel_invDepths, invdepth, sizeof(float) * width * height, cudaMemcpyDeviceToDevice), debug);

    return std::make_tuple(num_rendered, (int)bucket_sum);
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R, int B,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* dc,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
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
	char* image_buffer,
	char* sample_buffer,
	const float* dL_dpix,
	const float* dL_invdepths,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepths,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_ddc,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool antialiasing,
	bool debug,
	int tile_size)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(image_buffer, width * height);
	SampleState sampleState = SampleState::fromChunk(sample_buffer, B, tile_size * tile_size);

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
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height, R, B,
		imgState.bucket_offsets,
		sampleState.bucket_to_tile,
		sampleState.T, sampleState.ar, sampleState.ard,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		imgState.max_contrib,
		imgState.pixel_colors,
		imgState.pixel_invDepths,
		dL_dpix,
		dL_invdepths,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepths), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		dc,
		shs,
		geomState.clamped,
		opacities,
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
		dL_dinvdepths,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_ddc,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing), debug)
}

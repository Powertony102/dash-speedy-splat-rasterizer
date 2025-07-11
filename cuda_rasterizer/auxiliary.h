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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

// ---- BMES (Bounded Midpoint Ellipse Scan) Code ---- //

// A struct to hold the state for the integer-based scan.
struct IntegerScanState
{
	long long A, B, C, F;
	long long d, d1, d2;
};

// Initializes the integer scanner state for a given row.
__device__ inline void initialize_midpoint_scanner(
	IntegerScanState& scanner,
	long long A, long long B, long long C, long long D, long long E, long long F,
	int x, int y)
{
	// F(x, y) = Ax^2 + 2Bxy + Cy^2 + Dx + Ey + F
	scanner.d = (long long)A * x * x + (long long)2 * B * x * y + (long long)C * y * y + (long long)D * x + (long long)E * y + F;
	// F(x+1, y) - F(x,y) = 2Ax + A + 2By
	scanner.d1 = (long long)2 * A * x + A + (long long)2 * B * y + D;
	// Second-order difference is constant
	scanner.d2 = (long long)2 * A;
}

// Scans along the x-axis to find the start and end pixels of the ellipse for a given y.
__device__ inline void find_x_extents(
	IntegerScanState& scanner,
	int x_start, int x_end,
	int& x_min, int& x_max)
{
	x_min = x_start;
	x_max = x_end;
	
	// Scan from left to right to find the first pixel inside the ellipse
	for (int x = x_start; x <= x_end; ++x)
	{
		if (scanner.d <= 0)
		{
			x_min = x;
			break;
		}
		scanner.d += scanner.d1;
		scanner.d1 += scanner.d2;
	}

	// Scan from right to left to find the last pixel inside the ellipse
	for (int x = x_end; x >= x_min; --x)
	{
		if (scanner.d <= 0)
		{
			x_max = x;
			break;
		}
		scanner.d -= scanner.d1;
		scanner.d1 -= scanner.d2;
	}
}

__device__ inline uint32_t processTiles_BMES(
    const float4 con_o, const float t, const float2 p,
    int2 rect_min, int2 rect_max,
    const dim3 grid,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    const int tile_size = 16)
{
    // The ellipse equation is A'(x-px)^2 + 2B'(x-px)(y-py) + C'(y-py)^2 <= 1
	// where A', 2B', C' are the conic parameters.
	// The `con_o` variable holds {A', B', C', opacity}. Note that the middle term in `con_o` is B', not 2B'.
	// This appears to be a bug in the original code's conic calculation, as the standard form uses 2B.
	// We will follow the standard formula: A(x-px)^2 + 2B(x-px)(y-py) + C(y-py)^2 - t <= 0
	// Here, we use the `con_o` components as A, B, C directly, and use the opacity-derived threshold `t`.
	float A_f = con_o.x;
	float B_f = con_o.y;
	float C_f = con_o.z;

	// To use integer arithmetic, we scale all coefficients.
	// The precision of the integer representation is crucial.
	// We choose a sufficiently large scaling factor to maintain accuracy.
	const long long scale_factor = 1ll << 16;

	long long A = (long long)(A_f * scale_factor);
    long long B = (long long)(B_f * scale_factor);
    long long C = (long long)(C_f * scale_factor);

    // Pre-calculate terms for the general quadratic form:
    // Ax^2 + 2Bxy + Cy^2 + Dx + Ey + F = 0
    long long D = (long long)(-2.0f * (A_f * p.x + B_f * p.y) * scale_factor);
    long long E = (long long)(-2.0f * (B_f * p.x + C_f * p.y) * scale_factor);
    long long F = (long long)((A_f * p.x * p.x + 2.0f * B_f * p.x * p.y + C_f * p.y * p.y - t) * scale_factor);


	uint32_t tiles_count = 0;
	int x_start_px = rect_min.x * tile_size;
	int x_end_px = rect_max.x * tile_size;

	for (int y_row = rect_min.y; y_row < rect_max.y; ++y_row)
	{
		int y_top_px = y_row * tile_size;
		int y_bottom_px = y_top_px + tile_size - 1;

		IntegerScanState scanner_top, scanner_bottom;
		initialize_midpoint_scanner(scanner_top, A, B, C, D, E, F, x_start_px, y_top_px);
		initialize_midpoint_scanner(scanner_bottom, A, B, C, D, E, F, x_start_px, y_bottom_px);

		int x_min_top, x_max_top, x_min_bottom, x_max_bottom;
		find_x_extents(scanner_top, x_start_px, x_end_px, x_min_top, x_max_top);
		find_x_extents(scanner_bottom, x_start_px, x_end_px, x_min_bottom, x_max_bottom);

		int x_min_px, x_max_px;
		bool top_valid = x_min_top <= x_max_top;
		bool bottom_valid = x_min_bottom <= x_max_bottom;

		if (!top_valid && !bottom_valid) continue;

		if (top_valid && bottom_valid) {
			x_min_px = min(x_min_top, x_min_bottom);
			x_max_px = max(x_max_top, x_max_bottom);
		} else if (top_valid) {
			x_min_px = x_min_top;
			x_max_px = x_max_top;
		} else { // bottom_valid
			x_min_px = x_min_bottom;
			x_max_px = x_max_bottom;
		}
		
		int x_min_tile = x_min_px / tile_size;
		int x_max_tile = x_max_px / tile_size;

        // Correctly handle the case where we are only counting tiles
        int tiles_in_row = (x_max_tile - x_min_tile + 1);
        if (tiles_in_row > 0)
        {
            tiles_count += tiles_in_row;
            if (gaussian_keys_unsorted != nullptr)
            {
                for (int x_col = x_min_tile; x_col <= x_max_tile; ++x_col)
                {
                    uint64_t key = (uint64_t)y_row * grid.x + x_col;
                    key <<= 32;
                    key |= *((uint32_t*)&depth);
                    gaussian_keys_unsorted[off] = key;
                    gaussian_values_unsorted[off] = idx;
                    off++;
                }
            }
        }
	}
	return tiles_count;
}


__device__ inline uint32_t duplicateToTilesTouched(
    const float2 p, const float4 con_o, const dim3 grid,
    uint32_t idx, uint32_t off, float depth,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    const int tile_size = 16
    )
{

    //  ---- SNUGBOX Code ---- //

    // Calculate discriminant
    float disc = con_o.y * con_o.y - con_o.x * con_o.z;

    // If ill-formed ellipse, return 0
    if (con_o.x <= 0 || con_o.z <= 0 || disc >= 0) {
        return 0;
    }

    // Threshold: opacity * Gaussian = 1 / 255
    float t = 2.0f * log(con_o.w * 255.0f);

    // Simplified bounding box calculation based on the radii of the ellipse,
    // avoiding the expensive computeEllipseIntersection function. This provides
    // a conservative but efficient bounding box for the BMES algorithm.
    float mid = 0.5f * (con_o.x + con_o.z);
    float det = con_o.x * con_o.z - con_o.y * con_o.y;
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

    float2 bbox_min = { p.x - radius, p.y - radius };
    float2 bbox_max = { p.x + radius, p.y + radius };


    // Rectangular tile extent of ellipse
    int2 rect_min = {
        max(0, min((int)grid.x, (int)(bbox_min.x / tile_size))),
        max(0, min((int)grid.y, (int)(bbox_min.y / tile_size)))
    };
    int2 rect_max = {
        max(0, min((int)grid.x, (int)(bbox_max.x / tile_size + 1))),
        max(0, min((int)grid.y, (int)(bbox_max.y / tile_size + 1)))
    };

    int y_span = rect_max.y - rect_min.y;
    int x_span = rect_max.x - rect_min.x;

    // If no tiles are touched, return 0
    if (y_span * x_span == 0) {
        return 0;
    }

	// This now calls the BMES implementation.
	return processTiles_BMES(
		con_o, t, p,
		rect_min, rect_max,
		grid,
		idx, off, depth,
		gaussian_keys_unsorted,
		gaussian_values_unsorted,
		tile_size
	);
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
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

	p_view = transformPoint4x3(p_orig, viewmatrix);

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	if (p_hom.w <= 0.0000001f) return false;

	if (p_view.z <= 0.2f)
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

__device__ __forceinline__ void duplicateToTilesTouched(
	const float2 p, float3 cov, const float4 con_o, const dim3 grid,
	uint32_t idx, uint32_t& off, float depth,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	uint32_t& tiles_count,
	const int tile_size = 16)
{
	if (gaussian_keys_unsorted == nullptr)
	{
		// This is the counting pass
		// cov = (sigma_xx, sigma_xy, sigma_yy)
		float inv_a = cov.x;
		float inv_b = cov.y;
		float inv_c = cov.z;

		// Inverse of covariance matrix
		float det = inv_a * inv_c - inv_b * inv_b;
		if (det == 0.0f) return;
		float det_inv = 1.0f / det;
		float a = inv_c * det_inv, b = -inv_b * det_inv, c = inv_a * det_inv;
		
		float t = -2.0f * logf(max(con_o.w, 1.0f / 255.0f));
		if (t <= 0.f) return;

		// Extent of ellipse
		float x_denom = a*c - b*b;
		if (fabsf(x_denom) < 1e-9) return;

		float y_dist = sqrtf(fmaxf(0.f, t * a / x_denom));
		float y_min = p.y - y_dist, y_max = p.y + y_dist;
		float x_at_ymin = p.x - (b/a) * (y_min - p.y), x_at_ymax = p.x - (b/a) * (y_max - p.y);
		float x_dist = sqrtf(fmaxf(0.f, t * c / x_denom));
		float x_min = p.x - x_dist, x_max = p.x + x_dist;
		float y_at_xmin = p.y - (b/c) * (x_min - p.x), y_at_xmax = p.y - (b/c) * (x_max - p.x);
		float2 center = p;
		float2 p_verts[4] = {{x_min, y_at_xmin}, {x_max, y_at_xmax}, {x_at_ymin, y_min}, {x_at_ymax, y_max}};
		float2 left_box_min = center, left_box_max = center, right_box_min = center, right_box_max = center;
		bool is_left_box_init = false, is_right_box_init = false;
		#pragma unroll
		for (int i = 0; i < 4; i++) {
			if (p_verts[i].x <= center.x) {
				if (!is_left_box_init) { left_box_min = p_verts[i]; left_box_max = p_verts[i]; is_left_box_init = true; }
				else { left_box_min.x = fminf(left_box_min.x, p_verts[i].x); left_box_min.y = fminf(left_box_min.y, p_verts[i].y); left_box_max.x = fmaxf(left_box_max.x, p_verts[i].x); left_box_max.y = fmaxf(left_box_max.y, p_verts[i].y); }
			}
			if (p_verts[i].x >= center.x) {
				if (!is_right_box_init) { right_box_min = p_verts[i]; right_box_max = p_verts[i]; is_right_box_init = true; }
				else { right_box_min.x = fminf(right_box_min.x, p_verts[i].x); right_box_min.y = fminf(right_box_min.y, p_verts[i].y); right_box_max.x = fmaxf(right_box_max.x, p_verts[i].x); right_box_max.y = fmaxf(right_box_max.y, p_verts[i].y); }
			}
		}
		left_box_min.x = fminf(left_box_min.x, center.x); left_box_min.y = fminf(left_box_min.y, center.y);
		left_box_max.x = fmaxf(left_box_max.x, center.x); left_box_max.y = fmaxf(left_box_max.y, center.y);
		right_box_min.x = fminf(right_box_min.x, center.x); right_box_min.y = fminf(right_box_min.y, center.y);
		right_box_max.x = fmaxf(right_box_max.x, center.x); right_box_max.y = fmaxf(right_box_max.y, center.y);
		
		// Skew-Adaptive Stretching
		float l0_left = left_box_max.x - left_box_min.x, l0_right = right_box_max.x - right_box_min.x;
		float theta = 0.5f * atan2f(2.0f * cov.y, cov.x - cov.z);
		float beta = 1.0f; float stretch_factor;
		float cos2theta = cosf(2.0f * theta); const float epsilon = 0.01f; 
		if (fabsf(cos2theta) < epsilon) { stretch_factor = 2.0f; }
		else { stretch_factor = 1.0f + beta * fabsf(cos2theta); }
		float delta_left = (stretch_factor - 1.0f) * l0_left, delta_right = (stretch_factor - 1.0f) * l0_right;
		left_box_min.x -= delta_left; right_box_max.x += delta_right;
		int2 rect_min = {(int)fmaxf(0.f, floorf(left_box_min.x / tile_size)), (int)fmaxf(0.f, floorf(fminf(left_box_min.y, right_box_min.y) / tile_size))};
		int2 rect_max = {(int)fminf((float)grid.x, ceilf(right_box_max.x / tile_size)), (int)fminf((float)grid.y, ceilf(fmaxf(left_box_max.y, right_box_max.y) / tile_size))};
		tiles_count = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
	}
	else
	{
		// This is the writing pass
		// Re-computation logic is exactly the same as in the counting pass
		float inv_a = cov.x, inv_b = cov.y, inv_c = cov.z;
		float det = inv_a * inv_c - inv_b * inv_b;
		if (det == 0.0f) return;
		float det_inv = 1.0f / det;
		float a = inv_c * det_inv, b = -inv_b * det_inv, c = inv_a * det_inv;
		float t = -2.0f * logf(max(con_o.w, 1.0f / 255.0f));
		if (t <= 0.f) return;
		float x_denom = a*c - b*b;
		if (fabsf(x_denom) < 1e-9) return;
		float y_dist = sqrtf(fmaxf(0.f, t * a / x_denom));
		float y_min = p.y - y_dist, y_max = p.y + y_dist;
		float x_at_ymin = p.x - (b/a) * (y_min - p.y), x_at_ymax = p.x - (b/a) * (y_max - p.y);
		float x_dist = sqrtf(fmaxf(0.f, t * c / x_denom));
		float x_min = p.x - x_dist, x_max = p.x + x_dist;
		float y_at_xmin = p.y - (b/c) * (x_min - p.x), y_at_xmax = p.y - (b/c) * (x_max - p.x);
		float2 center = p;
		float2 p_verts[4] = {{x_min, y_at_xmin}, {x_max, y_at_xmax}, {x_at_ymin, y_min}, {x_at_ymax, y_max}};
		float2 left_box_min = center, left_box_max = center, right_box_min = center, right_box_max = center;
		bool is_left_box_init = false, is_right_box_init = false;
		#pragma unroll
		for (int i = 0; i < 4; i++) {
			if (p_verts[i].x <= center.x) {
				if (!is_left_box_init) { left_box_min = p_verts[i]; left_box_max = p_verts[i]; is_left_box_init = true; }
				else { left_box_min.x = fminf(left_box_min.x, p_verts[i].x); left_box_min.y = fminf(left_box_min.y, p_verts[i].y); left_box_max.x = fmaxf(left_box_max.x, p_verts[i].x); left_box_max.y = fmaxf(left_box_max.y, p_verts[i].y); }
			}
			if (p_verts[i].x >= center.x) {
				if (!is_right_box_init) { right_box_min = p_verts[i]; right_box_max = p_verts[i]; is_right_box_init = true; }
				else { right_box_min.x = fminf(right_box_min.x, p_verts[i].x); right_box_min.y = fminf(right_box_min.y, p_verts[i].y); right_box_max.x = fmaxf(right_box_max.x, p_verts[i].x); right_box_max.y = fmaxf(right_box_max.y, p_verts[i].y); }
			}
		}
		left_box_min.x = fminf(left_box_min.x, center.x); left_box_min.y = fminf(left_box_min.y, center.y);
		left_box_max.x = fmaxf(left_box_max.x, center.x); left_box_max.y = fmaxf(left_box_max.y, center.y);
		right_box_min.x = fminf(right_box_min.x, center.x); right_box_min.y = fminf(right_box_min.y, center.y);
		right_box_max.x = fmaxf(right_box_max.x, center.x); right_box_max.y = fmaxf(right_box_max.y, center.y);
		float l0_left = left_box_max.x - left_box_min.x, l0_right = right_box_max.x - right_box_min.x;
		float theta = 0.5f * atan2f(2.0f * cov.y, cov.x - cov.z);
		float beta = 1.0f; float stretch_factor;
		float cos2theta = cosf(2.0f * theta); const float epsilon = 0.01f; 
		if (fabsf(cos2theta) < epsilon) { stretch_factor = 2.0f; }
		else { stretch_factor = 1.0f + beta * fabsf(cos2theta); }
		float delta_left = (stretch_factor - 1.0f) * l0_left, delta_right = (stretch_factor - 1.0f) * l0_right;
		left_box_min.x -= delta_left; right_box_max.x += delta_right;
		int2 rect_min = {(int)fmaxf(0.f, floorf(left_box_min.x / tile_size)), (int)fmaxf(0.f, floorf(fminf(left_box_min.y, right_box_min.y) / tile_size))};
		int2 rect_max = {(int)fminf((float)grid.x, ceilf(right_box_max.x / tile_size)), (int)fminf((float)grid.y, ceilf(fmaxf(left_box_max.y, right_box_max.y) / tile_size))};

		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint32_t tile_idx = y * grid.x + x;
				uint64_t key = ((uint64_t)tile_idx << 32) | (*(uint32_t*)&depth);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
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
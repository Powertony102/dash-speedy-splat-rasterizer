# DashSpeedy Gaussian Rasterizer

A high-performance CUDA-based differentiable Gaussian rasterizer optimized for real-time 3D Gaussian Splatting rendering. This project extends the original differential Gaussian rasterization with advanced performance optimizations and dynamic tile size management.

## Features

### 🚀 Performance Optimizations
- **Dynamic Tile Size Management**: Automatically selects optimal tile sizes based on image resolution for maximum performance
- **Adaptive Memory Allocation**: Efficient memory management with bucket-based sampling system
- **CUDA-Optimized Kernels**: Highly optimized forward and backward passes for real-time rendering
- **Anti-aliasing Support**: Built-in anti-aliasing capabilities for high-quality output

### 🎯 Core Capabilities
- **Differentiable Rendering**: Full gradient support for training and optimization
- **3D Gaussian Splatting**: Native support for 3D Gaussian primitives with spherical harmonics
- **Real-time Performance**: Optimized for interactive applications and real-time rendering
- **Flexible Input Formats**: Support for both precomputed colors and spherical harmonics

### 🔧 Advanced Features
- **Resolution-Adaptive Tiling**
- **Bucket-Based Sampling**: Efficient memory management with dynamic bucket allocation
- **Depth Buffer Support**: Integrated depth buffer for proper occlusion handling
- **Debug Mode**: Comprehensive debugging capabilities for development

## Installation

### Prerequisites
- CUDA 11.7 or higher
- PyTorch 1.12 or higher
- Python 3.8 or higher
- CMake 3.18 or higher

### Build Instructions

```bash
# Clone the repository
git clone <repository-url>
cd dash-speedy-splat-rasterizer

# Install the package
pip install -e .
```

## Performance Characteristics

### Tile Size Impact
- **$16 \times 16$ tiles**: Optimal for high-resolution rendering
- **$8 \times 8$ tiles**: Balanced performance for medium resolutions
- **$4 \times 4$ tiles**: Best performance for low-resolution rendering

### Memory Efficiency
- Dynamic bucket allocation reduces memory overhead
- Efficient shared memory usage in CUDA kernels
- Optimized buffer management for large-scale scenes

## Architecture

### Core Components
- **CUDA Rasterizer**: Main rendering engine with optimized kernels
- **Tile Management**: Dynamic tile grid generation and management
- **Bucket System**: Efficient memory allocation for sampling
- **Gradient Computation**: Full backward pass for training

### Key Optimizations
1. **Dynamic Tile Sizing**: Resolution-adaptive tile size selection
2. **Bucket-Based Sampling**: Efficient memory management
3. **Shared Memory Optimization**: Maximized CUDA shared memory usage
4. **Radix Sorting**: Fast Gaussian sorting by tile and depth

## License

This software is free for non-commercial, research and evaluation use under the terms of the LICENSE.md file.

For inquiries contact: xinzeli0802@outlook.com

## Acknowledgments

- Based on the original differential Gaussian rasterization work
- Optimized for real-time performance and memory efficiency
- Enhanced with dynamic tile size management and bucket-based sampling

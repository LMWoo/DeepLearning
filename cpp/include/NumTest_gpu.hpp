#pragma once

#include <stdio.h>
#include <cuda.h>

#define CUDA_DEBUG
#if defined(CUDA_DEBUG)
#define CUDA_CHECK(x) do { \
    (x); \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) \
    { \
        printf("cuda failure %s:%d '%s'\n", \
        __FILE__, __LINE__, \
        cudaGetErrorString(e)); \
        exit(0); \
    } \
} while(0)
#else
#define CUDA_CHECK(x) (x)
#endif

namespace nt_gpu
{
    void test_matrix_mul();
    double* get_gpu_matrix_mul_double(double* dev_out, const double* dev_lhs, const double* dev_rhs, 
        const size_t lhs_rows, const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols);

    double* get_gpu_data_double(size_t size, double* data);
    double* get_cpu_data_double(size_t size, double* dev_data);

    void cudaFree_double(double* dev_data);
}
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

namespace NumTest_gpu
{
    void test_dot_gpu();
    
    void copy_gpu_to_gpu(size_t size, double* out_dev_data, const double* in_dev_data);
    void copy_cpu_to_gpu(size_t size, double* dev_data, const double* data);
    void copy_gpu_to_cpu(size_t size, double* data, const double* dev_data);

    double* gpu_malloc(size_t size);
    void* cpu_malloc(size_t size);
    void gpu_free(double* dev_data);
    void cpu_free(void* data);

    void transpose_gpu(double* out_dev_data, const double* in_dev_data, const size_t in_rows, const size_t in_cols);
    double* matrix_dot_gpu(double* dev_out, const double* dev_lhs, const double* dev_rhs, 
        const size_t lhs_rows, const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols);
    double* add_gpu(double* dev_out, const double* dev_lhs, const double* dev_rhs, const size_t& rows, const size_t& cols);

    void tanh_gpu(double* out_dev_data, const double* in_dev_data, const size_t& rows, const size_t& cols);
    void exp_gpu(double* out_dev_data, const double* in_dev_data, const size_t& rows, const size_t& cols);
    void sum_div_gpu(double* dev_data, const size_t& size);
    //void div_gpu(double* dev_data, const double& div, const size_t& size);
}
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

namespace cppTensor_Vec3_gpu
{
    void matMul_gpu(double* out_dev_data, const double* lhs_dev_data, const double* rhs_dev_data, int dim_z, int lhs_dim_y, int lhs_dim_x, int rhs_dim_y, int rhs_dim_x);
    void permute_gpu(double* out_dev_data, const double* in_dev_data, int* out_shape, int* in_shape, int* out_zyx);
}

namespace cppTensor_gpu
{
    void test_matMul_gpu();
    
    void copy_gpu_to_gpu(size_t size, double* out_dev_data, const double* in_dev_data);
    void copy_cpu_to_gpu(size_t size, double* dev_data, const double* data);
    void copy_gpu_to_cpu(size_t size, double* data, const double* dev_data);

    double* gpu_malloc(size_t size);
    void* cpu_malloc(size_t size);
    void gpu_free(double* dev_data);
    void cpu_free(void* data);

    void transpose_gpu(double* out_dev_data, const double* in_dev_data, const size_t in_rows, const size_t in_cols);
    void matMul_gpu(double* dev_out, const double* dev_lhs, const double* dev_rhs, 
        const size_t lhs_rows, const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols, bool useSharedMemory);
    void transpose_matMul_gpu(double* dev_out, const double* dev_lhs, const double* dev_rhs,
        const size_t lhs_rows, const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols);
    
    void add_gpu(double* dev_out, const double* dev_lhs, const double* dev_rhs, const size_t size);
    void zeros_gpu(double* dev_data, const size_t size);

    void tanh_gpu(double* out_dev_data, const double* in_dev_data, const size_t& rows, const size_t& cols);
    void exp_gpu(double* out_dev_data, const double* in_dev_data, const size_t& rows, const size_t& cols);
    void sum_div_gpu(double* dev_data, const size_t& size);
    void minus_gpu(double* dev_data, const size_t& size);
    void log_gpu(double* dev_data, const size_t& size);
    void deriv_softmax_gpu(size_t size, double* out_dY_data, double* out_loss_data, const double* in_Y_data, double* labels);
    void mul_gpu(double* out_dev_data, double inValue, const size_t& size);
    void mul_gpu(double* out_dev_data, const double* lhs_dev_data, const double* rhs_dev_data, const size_t& size);
    void deriv_tanh_gpu(double* out_dev_data, const double* in_dev_data, const size_t& size);
    void clip_gpu(double* out_dev_data, double low, double high, const size_t size);
    void optimizer_gpu(double* param, double* mem, const double* dparam, const size_t size);

    //void div_gpu(double* dev_data, const double& div, const size_t& size);
}
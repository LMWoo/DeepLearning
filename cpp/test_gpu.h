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

namespace test_gpu
{
    void test();
    void test_matrix_add();
    void test_matrix_mul();
}
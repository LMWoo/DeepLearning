#include "NumTest_gpu.hpp"

namespace nt_gpu
{
    __global__ void mulKernel(double *c, const double *a, const double *b, const int WIDTH)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int i = y * WIDTH + x;

        double sum = 0.0f;
        for (int k = 0; k < WIDTH; ++k)
        {
            double lhs = a[y * WIDTH + k];
            double rhs = b[k * WIDTH + x];
            sum += lhs * rhs;
        }
        c[i] = sum;
    }

    __global__ void mulKernel(double *dev_out, const double* dev_lhs, const double* dev_rhs, 
        const size_t lhs_rows, const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols)
    {
        size_t x = threadIdx.x;
        size_t y = threadIdx.y;
        size_t i = y * rhs_cols + x;

        double sum = 0.0;

        for (size_t k = 0; k < lhs_cols; ++k)
        {
            sum += dev_lhs[y * lhs_cols + k] * dev_rhs[k * rhs_cols + x];
        }

        dev_out[i] = sum;
    }

    void copy_cpu_to_gpu(size_t size, double* dev_data, const double* data)
    {
        CUDA_CHECK(cudaMemcpy(dev_data, data, size, cudaMemcpyHostToDevice));
    }

    void copy_gpu_to_cpu(size_t size, double* data, const double* dev_data)
    {
        CUDA_CHECK(cudaMemcpy(data, dev_data, size, cudaMemcpyDeviceToHost));
    }

    double* gpu_matrix_mul_double(double* dev_out, const double* dev_lhs, const double* dev_rhs, 
        const size_t lhs_rows, const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols)
    {
        if (dev_out == nullptr)
        {
            return nullptr;
        }

        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(rhs_cols, lhs_rows, 1);

        mulKernel<<<dimGrid, dimBlock>>>(dev_out, dev_lhs, dev_rhs, lhs_rows, lhs_cols, rhs_rows, rhs_cols);
        return dev_out;
    }

    double* gpu_malloc(size_t size)
    {
        double* dev_data=nullptr;
        CUDA_CHECK(cudaMalloc((void**)&dev_data, size));
        return dev_data;
    }

    void* cpu_malloc(size_t size)
    {
        return malloc(size);
    }

    void gpu_free(double* dev_data)
    {
        if (dev_data)
        {
            CUDA_CHECK(cudaFree(dev_data));
        }
    }

    void cpu_free(void* data)
    {
        if (data)
        {
            free(data);
        }
    }
    
    void test_matrix_mul()
    {
        int WIDTH = 512;
        int TILE_WIDTH = 16;
        int GRID_WIDTH = WIDTH / TILE_WIDTH;

        double a[WIDTH][WIDTH];
        double b[WIDTH][WIDTH];
        double c[WIDTH][WIDTH] = {0};

        for (int y = 0; y < WIDTH; ++y)
        {
            for (int x = 0; x < WIDTH; ++x)
            {
                // a[y][x] = y * 10 + x;
                // b[y][x] = a[y][x] * 100;
                a[y][x] = 1.0;
                b[y][x] = 1.0;
            }
        }

        double *dev_a = 0;
        double *dev_b = 0;
        double *dev_c = 0;
        CUDA_CHECK(cudaMalloc((void**)&dev_a, WIDTH * WIDTH * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&dev_b, WIDTH * WIDTH * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&dev_c, WIDTH * WIDTH * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(dev_a, a, WIDTH * WIDTH * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_b, b, WIDTH * WIDTH * sizeof(double), cudaMemcpyHostToDevice));

        dim3 dimGrid(GRID_WIDTH, GRID_WIDTH, 1);
        dim3 dimThread(TILE_WIDTH, TILE_WIDTH, 1);
        for (int i = 0; i < 128; ++i)
            mulKernel<<<dimGrid, dimThread>>>(dev_c, dev_a, dev_b, WIDTH);

        CUDA_CHECK(cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);

        // int print_width = WIDTH > 64 ? 64 : WIDTH;
        // for (int y = 0; y < print_width; ++y)
        // {
        //     for (int x = 0; x < print_width; ++x)
        //     {
        //         printf("%lf ", c[y][x]);
        //     }
        //     printf("\n");
        // }



        // // const int WIDTH = 5;
        // int a[WIDTH][WIDTH];
        // int b[WIDTH][WIDTH];
        // int c[WIDTH][WIDTH] = {0};

        // for (int y = 0; y < WIDTH; ++y)
        // {
        //     for (int x = 0; x < WIDTH; ++x)
        //     {
        //         // a[y][x] = y * 10 + x;
        //         // b[y][x] = a[y][x] * 100;
        //         a[y][x] = 1;
        //         b[y][x] = 1;
        //     }
        // }

        // int *dev_a = 0;
        // int *dev_b = 0;
        // int *dev_c = 0;
        // CUDA_CHECK(cudaMalloc((void**)&dev_a, WIDTH * WIDTH * sizeof(int)));
        // CUDA_CHECK(cudaMalloc((void**)&dev_b, WIDTH * WIDTH * sizeof(int)));
        // CUDA_CHECK(cudaMalloc((void**)&dev_c, WIDTH * WIDTH * sizeof(int)));

        // CUDA_CHECK(cudaMemcpy(dev_a, a, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(dev_b, b, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice));

        // dim3 dimGrid(1, 1, 1);
        // dim3 dimBlock(WIDTH, WIDTH, 1);
        // mulKernel<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, WIDTH);

        // CUDA_CHECK(cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost));
        // cudaFree(dev_c);
        // cudaFree(dev_a);
        // cudaFree(dev_b);

        // for (int y = 0; y < WIDTH; ++y)
        // {
        //     for (int x = 0; x < WIDTH; ++x)
        //     {
        //         printf("%d ", c[y][x]);
        //     }
        //     printf("\n");
        // }
    }
}
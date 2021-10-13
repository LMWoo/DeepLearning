#include "test_gpu.h"

namespace test_gpu
{
    void test()
    {
        const int SIZE = 5;
        const int a[SIZE] = {1, 2, 3, 4, 5};
        int b[SIZE] = {0, 0, 0, 0, 0};

        printf("a = ");
        for (int i = 0; i < 5; ++i)
        {
            printf("%d ", a[i]);
        }
        printf("\n");

        int *dev_a = 0;
        int *dev_b = 0;

        CUDA_CHECK(cudaMalloc((void**)&dev_a, SIZE * sizeof(int)));
        cudaMalloc((void**)&dev_b, SIZE * sizeof(int));

        cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, dev_a, SIZE * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(b, dev_b, SIZE * sizeof(int), cudaMemcpyDeviceToHost);


        printf("b = ");
        for (int i = 0; i < 5; ++i)
        {
            printf("%d ", b[i]);
        }
        printf("\n");
    }

    __global__ void addKernel(int *c, const int *a, const int *b)
    {
        int x = threadIdx.x;
        int y = threadIdx.y;

        int i = y * (blockDim.x) + x;
        c[i] = a[i] + b[i];
    }

    void test_matrix_add()
    {
        const int WIDTH = 5;
        int a[WIDTH][WIDTH];
        int b[WIDTH][WIDTH];
        int c[WIDTH][WIDTH] = {0};

        for (int y = 0; y < WIDTH; ++y)
        {
            for (int x = 0; x < WIDTH; ++x)
            {
                a[y][x] = y * 10 + x;
                b[y][x] = a[y][x] * 100;
            }
        }

        int *dev_a = 0;
        int *dev_b = 0;
        int *dev_c = 0;
        CUDA_CHECK(cudaMalloc((void**)&dev_a, WIDTH * WIDTH * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&dev_b, WIDTH * WIDTH * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&dev_c, WIDTH * WIDTH * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(dev_a, a, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_b, b, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice));

        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(WIDTH, WIDTH, 1);
        addKernel<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b);    

        CUDA_CHECK(cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost));
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);

        for (int y = 0; y < WIDTH; ++y)
        {
            for (int x = 0; x < WIDTH; ++x)
            {
                printf("%d ", c[y][x]);
            }
            printf("\n");
        }
    }
}
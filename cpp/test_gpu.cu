#include "test_gpu.h"
#include <cuda.h>

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

        cudaMalloc((void**)&dev_a, SIZE * sizeof(int));
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
}
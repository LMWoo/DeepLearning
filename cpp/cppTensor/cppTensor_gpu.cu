#include "cppTensor_gpu.hpp"

#define TILE_WIDTH 8
#define VEC3_TILE_SIZE 4

namespace cppTensor_Vec3_gpu
{
    __device__ int getIdx(int z, int y, int x, int dim_y, int dim_x)
    {
        return x + y * dim_x + z * dim_y * dim_x;
    }

    __global__ void matMul_(double* out_dev_data, const double* lhs_dev_data, const double* rhs_dev_data, int lhs_dim_y, int lhs_dim_x, int rhs_dim_y, int rhs_dim_x)
    {
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        for (int i = 0; i < lhs_dim_y; ++i)
        {
            for (int j = 0; j < rhs_dim_x; ++j)
            {
                double sum = 0.0;

                for (int k = 0; k < lhs_dim_x; ++k)
                {
                    sum += lhs_dev_data[getIdx(z, i, k, lhs_dim_y, lhs_dim_x)] * rhs_dev_data[getIdx(z, k, j, rhs_dim_y, rhs_dim_x)];
                }

                out_dev_data[getIdx(z, i, j, lhs_dim_y, rhs_dim_x)] = sum;
            }
        }
    }

    void matMul_gpu(double* out_dev_data, const double* lhs_dev_data, const double* rhs_dev_data, int dim_z, int lhs_dim_y, int lhs_dim_x, int rhs_dim_y, int rhs_dim_x)
    {
        int grid_x = rhs_dim_x / VEC3_TILE_SIZE;
        int grid_y = lhs_dim_y / VEC3_TILE_SIZE;
        int grid_z = dim_z / VEC3_TILE_SIZE;
        grid_x += rhs_dim_x % VEC3_TILE_SIZE ? 1 : 0;
        grid_y += lhs_dim_y % VEC3_TILE_SIZE ? 1 : 0;
        grid_z += dim_z % VEC3_TILE_SIZE ? 1 : 0;

        dim3 dimGrid(grid_x, grid_y, grid_z);
        dim3 dimBlock(VEC3_TILE_SIZE, VEC3_TILE_SIZE, VEC3_TILE_SIZE);

        matMul_<<<dimGrid, dimBlock>>>(out_dev_data, lhs_dev_data, rhs_dev_data, lhs_dim_y, lhs_dim_x, rhs_dim_y, rhs_dim_x);
    }

    __global__ void permute_(double* out_dev_data, const double* in_dev_data, 
        int out_dim_z, int out_dim_y, int out_dim_x, int in_dim_z, int in_dim_y, int in_dim_x, int out_z, int out_y, int out_x)
    {
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int zyx[3] = {z, y, x};

        if ((z >= in_dim_z || y >= in_dim_y) || x >= in_dim_x)
        {
            return;
        }

        out_dev_data[getIdx(zyx[out_z], zyx[out_y], zyx[out_x], out_dim_y, out_dim_x)] = in_dev_data[getIdx(z, y, x, in_dim_y, in_dim_x)];
    }

    void permute_gpu(double* out_dev_data, const double* in_dev_data, int* out_shape, int* in_shape, int* out_zyx)
    {
        int grid_x = in_shape[2] / VEC3_TILE_SIZE;
        int grid_y = in_shape[1] / VEC3_TILE_SIZE;
        int grid_z = in_shape[0] / VEC3_TILE_SIZE;
        grid_x += in_shape[2] % VEC3_TILE_SIZE ? 1 : 0;
        grid_y += in_shape[1] % VEC3_TILE_SIZE ? 1 : 0;
        grid_z += in_shape[0] % VEC3_TILE_SIZE ? 1 : 0;

        dim3 dimGrid(grid_x, grid_y, grid_z);
        dim3 dimBlock(VEC3_TILE_SIZE, VEC3_TILE_SIZE, VEC3_TILE_SIZE);

        permute_<<<dimGrid, dimBlock>>>(out_dev_data, in_dev_data, 
            out_shape[0], out_shape[1], out_shape[2], in_shape[0], in_shape[1], in_shape[2], out_zyx[0], out_zyx[1], out_zyx[2]);
    }
}

namespace cppTensor_gpu
{
    __global__ void test_matMul(double *c, const double *a, const double *b, const int WIDTH)
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
    
    void test_matMul_gpu()
    {
        int TEST_WIDTH = 512;
        int TEST_TILE_WIDTH = 16;
        int TEST_GRID_WIDTH = TEST_WIDTH / TEST_TILE_WIDTH;

        double a[TEST_WIDTH][TEST_WIDTH];
        double b[TEST_WIDTH][TEST_WIDTH];
        double c[TEST_WIDTH][TEST_WIDTH] = {0};

        for (int y = 0; y < TEST_WIDTH; ++y)
        {
            for (int x = 0; x < TEST_WIDTH; ++x)
            {
                a[y][x] = 1.0;
                b[y][x] = 1.0;
            }
        }

        double *dev_a = 0;
        double *dev_b = 0;
        double *dev_c = 0;
        CUDA_CHECK(cudaMalloc((void**)&dev_a, TEST_WIDTH * TEST_WIDTH * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&dev_b, TEST_WIDTH * TEST_WIDTH * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&dev_c, TEST_WIDTH * TEST_WIDTH * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(dev_a, a, TEST_WIDTH * TEST_WIDTH * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_b, b, TEST_WIDTH * TEST_WIDTH * sizeof(double), cudaMemcpyHostToDevice));

        dim3 dimGrid(TEST_GRID_WIDTH, TEST_GRID_WIDTH, 1);
        dim3 dimThread(TEST_TILE_WIDTH, TEST_TILE_WIDTH, 1);
        for (int i = 0; i < 128; ++i)
            test_matMul<<<dimGrid, dimThread>>>(dev_c, dev_a, dev_b, TEST_WIDTH);

        CUDA_CHECK(cudaMemcpy(c, dev_c, TEST_WIDTH * TEST_WIDTH * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);

        for (int y = 0; y < TEST_WIDTH; ++y)
        {
            for (int x = 0; x < TEST_WIDTH; ++x)
            {
                printf("%lf ", c[y][x]);
            }
            printf("\n");
        }
    }

    void copy_gpu_to_gpu(size_t size, double* out_dev_data, const double* in_dev_data)
    {
        CUDA_CHECK(cudaMemcpy(out_dev_data, in_dev_data, size, cudaMemcpyDeviceToDevice));
    }

    void copy_cpu_to_gpu(size_t size, double* dev_data, const double* data)
    {
        CUDA_CHECK(cudaMemcpy(dev_data, data, size, cudaMemcpyHostToDevice));
    }

    void copy_gpu_to_cpu(size_t size, double* data, const double* dev_data)
    {
        CUDA_CHECK(cudaMemcpy(data, dev_data, size, cudaMemcpyDeviceToHost));
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

    __global__ void transpose(double* out_dev_data, const double* in_dev_data, const size_t in_rows, const size_t in_cols)
    {
        size_t in_x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t in_y = blockIdx.y * blockDim.y + threadIdx.y;
        if (in_x >= in_cols || in_y >= in_rows)
        {
            return;
        }

        out_dev_data[in_x * in_rows + in_y] = in_dev_data[in_y * in_cols + in_x];
    }

    void transpose_gpu(double* out_dev_data, const double* in_dev_data, const size_t in_rows, const size_t in_cols)
    {
        dim3 dimGrid(in_cols / TILE_WIDTH + 1, in_rows / TILE_WIDTH + 1, 1);
        dim3 dimThread(TILE_WIDTH, TILE_WIDTH, 1);

        transpose<<<dimGrid, dimThread>>>(out_dev_data, in_dev_data, in_rows, in_cols);
    }

    __global__ void matMul_(double *dev_out, const double* dev_lhs, const double* dev_rhs, 
        const size_t lhs_rows, const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= rhs_cols || y >= lhs_rows)
        {
            return;
        }

        size_t i = y * rhs_cols + x;

        double sum = 0.0;

        for (size_t k = 0; k < lhs_cols; ++k)
        {
            sum += dev_lhs[y * lhs_cols + k] * dev_rhs[k * rhs_cols + x];
        }

        dev_out[i] = sum;
    }

    __global__ void matMul_sharedMemory_(double *dev_out, const double* dev_lhs, const double* dev_rhs, 
        const size_t lhs_rows, const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;

        __shared__ double s_lhs[TILE_WIDTH][TILE_WIDTH];
        __shared__ double s_rhs[TILE_WIDTH][TILE_WIDTH];

        double sum = 0.0;

        size_t lhs_y = 0;
        size_t lhs_x = 0;
        size_t rhs_y = 0;
        size_t rhs_x = 0;

        int m = lhs_cols / TILE_WIDTH;
        m += lhs_cols % TILE_WIDTH ? 1 : 0;
        for (int i = 0; i < m; ++i)
        {
            lhs_y = y;
            lhs_x = i * TILE_WIDTH + threadIdx.x;
            rhs_y = i * TILE_WIDTH + threadIdx.y;
            rhs_x = x;

            if (lhs_y >= lhs_rows || lhs_x >= lhs_cols)
            {
                s_lhs[threadIdx.y][threadIdx.x] = 0.0;    
            }
            else
            {
                s_lhs[threadIdx.y][threadIdx.x] = dev_lhs[lhs_y * lhs_cols + lhs_x];
            }

            if (rhs_y >= rhs_rows || rhs_x >= rhs_cols)
            {
                s_rhs[threadIdx.y][threadIdx.x] = 0.0;
            }
            else
            {
                s_rhs[threadIdx.y][threadIdx.x] = dev_rhs[rhs_y * rhs_cols + rhs_x];
            }

            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k)
            {
                sum += s_lhs[threadIdx.y][k] * s_rhs[k][threadIdx.x];
            }
        }

        __syncthreads();


        if (x >= rhs_cols || y >= lhs_rows)
        {
            return;
        }

        dev_out[y * rhs_cols + x] = sum;
    }

    void matMul_gpu(double* dev_out, const double* dev_lhs, const double* dev_rhs, 
        const size_t lhs_rows, const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols, bool useSharedMemory)
    {
        int blockDimY = lhs_rows / TILE_WIDTH;
        int blockDimX = rhs_cols / TILE_WIDTH;
        blockDimY += lhs_rows % TILE_WIDTH ? 1 : 0;
        blockDimX += rhs_cols % TILE_WIDTH ? 1 : 0;

        dim3 dimGrid(blockDimX, blockDimY, 1);
        dim3 dimThread(TILE_WIDTH, TILE_WIDTH, 1);

        if (useSharedMemory)
        {
            matMul_sharedMemory_<<<dimGrid, dimThread>>>(dev_out, dev_lhs, dev_rhs, lhs_rows, lhs_cols, rhs_rows, rhs_cols);
        }
        else
        {
            matMul_<<<dimGrid, dimThread>>>(dev_out, dev_lhs, dev_rhs, lhs_rows, lhs_cols, rhs_rows, rhs_cols);
        }
    }

    __global__ void transpose_matMul_(double *dev_out, const double* dev_lhs, const double* dev_rhs, 
        const size_t lhs_rows, const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= rhs_rows || y >= lhs_rows)
        {
            return;
        }

        size_t i = y * rhs_rows + x;

        double sum = 0.0;

        for (size_t k = 0; k < lhs_cols; ++k)
        {
            sum += dev_lhs[y * lhs_cols + k] * dev_rhs[x * rhs_cols + k];
        }

        dev_out[i] = sum;
    }

    void transpose_matMul_gpu(double* dev_out, const double* dev_lhs, const double* dev_rhs,
        const size_t lhs_rows, const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols)
    {
        int blockDimY = lhs_rows / TILE_WIDTH;
        int blockDimX = rhs_rows / TILE_WIDTH;

        blockDimY += lhs_rows % TILE_WIDTH ? 1 : 0;
        blockDimX += rhs_rows % TILE_WIDTH ? 1 : 0;

        dim3 dimGrid(blockDimX, blockDimY, 1);
        dim3 dimThread(TILE_WIDTH, TILE_WIDTH, 1);   

        transpose_matMul_<<<dimGrid, dimThread>>>(dev_out, dev_lhs, dev_rhs, lhs_rows, lhs_cols, rhs_rows, rhs_cols);
    }

    __global__ void add_(double* dev_out, const double* dev_lhs, const double* dev_rhs, const size_t size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= size)
        {
            return;
        }
        
        dev_out[i] = dev_lhs[i] + dev_rhs[i];
    }

    void add_gpu(double* dev_out, const double* dev_lhs, const double* dev_rhs, const size_t size)
    {
        dim3 dimGrid(size / TILE_WIDTH + 1, 1, 1);
        dim3 dimBlock(TILE_WIDTH, 1, 1);

        add_<<<dimGrid, dimBlock>>>(dev_out, dev_lhs, dev_rhs, size);
    }

    __global__ void zeros_(double* dev_data, const size_t size)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= size)
        {
            return;
        }
        dev_data[x] = 0.0;
    }

    void zeros_gpu(double* dev_data, const size_t size)
    {
        // dim3 dimGrid(size / TILE_WIDTH + 1, 1, 1);
        // dim3 dimBlock(TILE_WIDTH, 1, 1);

        // zeros_<<<dimGrid, dimBlock>>>(dev_data, size);
        CUDA_CHECK(cudaMemset(dev_data, 0, size));
    }

    __global__ void tanh_(double* out_dev_data, const double* in_dev_data)
    {
        size_t i = threadIdx.y * blockDim.x + threadIdx.x;
        out_dev_data[i] = tanh(in_dev_data[i]);
    }

    void tanh_gpu(double* out_dev_data, const double* in_dev_data, const size_t& rows, const size_t& cols)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(cols, rows, 1);

        tanh_<<<dimGrid, dimBlock>>>(out_dev_data, in_dev_data);
    }

    __global__ void exp_(double* out_dev_data, const double* in_dev_data)
    {
        size_t i = threadIdx.y * blockDim.x + threadIdx.x;
        out_dev_data[i] = exp(in_dev_data[i]);
    }

    void exp_gpu(double* out_dev_data, const double* in_dev_data, const size_t& rows, const size_t& cols)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(cols, rows, 1);

        exp_<<<dimGrid, dimBlock>>>(out_dev_data, in_dev_data);
    }

    __global__ void sum_div_(double* dev_data)
    {
        __shared__ double arr[1024];
        
        arr[threadIdx.x] = dev_data[threadIdx.x];

        __syncthreads();

        for (size_t stride = 1; stride <= blockDim.x; stride *= 2)
        {
            if (threadIdx.x % stride == 0)
            {
                arr[2 * threadIdx.x] += arr[2 * threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        dev_data[threadIdx.x] /= arr[0];
    }

    void sum_div_gpu(double* dev_data, const size_t& size)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(size, 1, 1);

        sum_div_<<<dimGrid, dimBlock>>>(dev_data);
    }

    __global__ void minus_(double* dev_data)
    {
        dev_data[threadIdx.x] = -dev_data[threadIdx.x];
    }

    void minus_gpu(double* dev_data, const size_t& size)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(size, 1, 1);

        minus_<<<dimGrid, dimBlock>>>(dev_data);
    }

    __global__ void log_(double* dev_data)
    {
        dev_data[threadIdx.x] = log(dev_data[threadIdx.x]);
    }

    void log_gpu(double* dev_data, const size_t& size)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(size, 1, 1);

        log_<<<dimGrid, dimBlock>>>(dev_data);
    }


    __global__ void deriv_softmax_(double* out_dY_data, double* out_loss_data, const double* in_Y_data, double* labels)
    {
        out_loss_data[(size_t)labels[0]] = in_Y_data[(size_t)labels[0]];
        out_dY_data[(size_t)labels[0]] -= 1.0;
    }

    void deriv_softmax_gpu(size_t size, double* out_dY_data, double* out_loss_data, const double* in_Y_data, double* labels)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(size, 1, 1);

        deriv_softmax_<<<dimGrid, dimBlock>>>(out_dY_data, out_loss_data, in_Y_data, labels);
    }

    __global__ void mul_(double* out_dev_data, double inValue)
    {
        out_dev_data[threadIdx.x] = out_dev_data[threadIdx.x] * inValue;
    }

    void mul_gpu(double* out_dev_data, double inValue, const size_t& size)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(size, 1, 1);

        mul_<<<dimGrid, dimBlock>>>(out_dev_data, inValue);
    }

    __global__ void mul_(double* out_dev_data, const double* lhs_dev_data, const double* rhs_dev_data)
    {
        out_dev_data[threadIdx.x] = lhs_dev_data[threadIdx.x] * rhs_dev_data[threadIdx.x];
    }

    void mul_gpu(double* out_dev_data, const double* lhs_dev_data, const double* rhs_dev_data, const size_t& size)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(size, 1, 1);

        mul_<<<dimGrid, dimBlock>>>(out_dev_data, lhs_dev_data, rhs_dev_data);
    }

    __global__ void deriv_tanh_(double* out_dev_data, const double* in_dev_data)
    {
        out_dev_data[threadIdx.x] = 1.0 - in_dev_data[threadIdx.x] * in_dev_data[threadIdx.x];
    }

    void deriv_tanh_gpu(double* out_dev_data, const double* in_dev_data, const size_t& size)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(size, 1, 1);

        deriv_tanh_<<<dimGrid, dimBlock>>>(out_dev_data, in_dev_data);
    }
    
    __global__ void clip_(double* out_dev_data, double low, double high, const size_t size)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= size)
        {
            return;
        }

        out_dev_data[x] = min(high, out_dev_data[x]);
        out_dev_data[x] = max(low, out_dev_data[x]);
    }

    void clip_gpu(double* out_dev_data, double low, double high, const size_t size)
    {
        dim3 dimGrid(size / TILE_WIDTH + 1, 1, 1);
        dim3 dimBlock(TILE_WIDTH, 1, 1);

        clip_<<<dimGrid, dimBlock>>>(out_dev_data, low, high, size);
    }

    __global__ void optimizer_(double* param, double* mem, const double* dparam, const size_t size)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= size)
        {
            return;
        }
        mem[x] += dparam[x] * dparam[x];
        param[x] += (-0.01 * dparam[x]) / sqrt(mem[x] + 1e-8);
    }

    void optimizer_gpu(double* param, double* mem, const double* dparam, const size_t size)
    {
        dim3 dimGrid(size / TILE_WIDTH + 1, 1, 1);
        dim3 dimBlock(TILE_WIDTH, 1, 1);

        optimizer_<<<dimGrid, dimBlock>>>(param, mem, dparam, size);
    }

    // __global__ void div_(double* dev_data, const double& div)
    // {
    //     dev_data[threadIdx.x] /= div;
    // }

    // void div_gpu(double* dev_data, const double& div, const size_t& size)
    // {
    //     dim3 dimGrid(1, 1, 1);
    //     dim3 dimBlock(size, 1, 1);

    //     div_<<<dimGrid, dimBlock>>>(dev_data, div);
    // }
}
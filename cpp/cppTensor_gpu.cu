#include "cppTensor_gpu.hpp"

namespace cppTensor_gpu
{
    __global__ void test_dot(double *c, const double *a, const double *b, const int WIDTH)
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
    
    void test_dot_gpu()
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
            test_dot<<<dimGrid, dimThread>>>(dev_c, dev_a, dev_b, WIDTH);

        CUDA_CHECK(cudaMemcpy(c, dev_c, WIDTH * WIDTH * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);

        for (int y = 0; y < WIDTH; ++y)
        {
            for (int x = 0; x < WIDTH; ++x)
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
        size_t x = threadIdx.x;
        size_t y = threadIdx.y;

        out_dev_data[x * in_rows + y] = in_dev_data[y * in_cols + x];
    }

    void transpose_gpu(double* out_dev_data, const double* in_dev_data, const size_t in_rows, const size_t in_cols)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimThread(in_cols, in_rows, 1);

        transpose<<<dimGrid, dimThread>>>(out_dev_data, in_dev_data, in_rows, in_cols);
    }

    __global__ void matrix_dot(double *dev_out, const double* dev_lhs, const double* dev_rhs, 
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

    double* matrix_dot_gpu(double* dev_out, const double* dev_lhs, const double* dev_rhs, 
        const size_t lhs_rows, const size_t lhs_cols, const size_t rhs_rows, const size_t rhs_cols)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimThread(rhs_cols, lhs_rows, 1);

        matrix_dot<<<dimGrid, dimThread>>>(dev_out, dev_lhs, dev_rhs, lhs_rows, lhs_cols, rhs_rows, rhs_cols);
        return dev_out;
    }

    __global__ void add(double* dev_out, const double* dev_lhs, const double* dev_rhs)
    {
        size_t i = threadIdx.y * blockDim.x + threadIdx.x;
        dev_out[i] = dev_lhs[i] + dev_rhs[i];
    }

    double* add_gpu(double* dev_out, const double* dev_lhs, const double* dev_rhs, const size_t& rows, const size_t& cols)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(cols, rows, 1);

        add<<<dimGrid, dimBlock>>>(dev_out, dev_lhs, dev_rhs);
        return dev_out;
    }

    __global__ void zeros_(double* dev_data)
    {
        dev_data[threadIdx.x] = 0.0;
    }

    void zeros_gpu(double* dev_data, const size_t& size)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(size, 1, 1);

        zeros_<<<dimGrid, dimBlock>>>(dev_data);
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
    
    __global__ void clip_(double* out_dev_data, double low, double high)
    {
        out_dev_data[threadIdx.x] = min(high, out_dev_data[threadIdx.x]);
        out_dev_data[threadIdx.x] = max(low, out_dev_data[threadIdx.x]);
    }

    void clip_gpu(double* out_dev_data, double low, double high, const size_t& size)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(size, 1, 1);

        clip_<<<dimGrid, dimBlock>>>(out_dev_data, low, high);
    }

    __global__ void optimizer_(double* param, double* mem, const double* dparam)
    {
        mem[threadIdx.x] += dparam[threadIdx.x] * dparam[threadIdx.x];
        param[threadIdx.x] += (-0.01 * dparam[threadIdx.x]) / sqrt(mem[threadIdx.x] + 1e-8);
    }

    void optimizer_gpu(double* param, double* mem, const double* dparam, const size_t& size)
    {
        dim3 dimGrid(1, 1, 1);
        dim3 dimBlock(size, 1, 1);

        optimizer_<<<dimGrid, dimBlock>>>(param, mem, dparam);
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
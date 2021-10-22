#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <cuda.h>
#include "NumTest_gpu.hpp"

// #define NUMTEST_DEBUG

#if defined(NUMTEST_DEBUG)
#define PRINT_DEBUG(str, ...) do { \
    printf((str), ##__VA_ARGS__); \
} while(0)
#else
#define PRINT_DEBUG(str, ...)
#endif

template<typename dtype>
class numTest
{
public:
    using numpyArray = pybind11::array_t<dtype, pybind11::array::c_style>;
    using numpyArrayGeneric = pybind11::array;

private:
    class shape
    {
    public:
        size_t rows{0};
        size_t cols{0};
        
        size_t size() const
        {
            return rows * cols;
        }

        void print()
        {
            printf("shape rows %d cols %d\n", rows, cols);
        }
        shape() = default;

        shape(size_t rows, size_t cols)
        {
            this->rows = rows;
            this->cols = cols;
        }
    };
public:
    dtype* data_{nullptr};
    dtype* dev_data_{nullptr};
    shape shape_;
    bool is_cuda_{false};

private:
    void newArray()
    {
        deleteArray();
        data_ = (dtype*)malloc(sizeof(dtype) * shape_.size());
        this->zeros();

        print_pointer("newArray()");
    }

    void newArray(size_t rows, size_t cols)
    {
        shape_ = shape(rows, cols);
        newArray();
    }

    void deleteArray()
    {
        print_pointer("deleteArray()");

        nt_gpu::cpu_free(data_);
        data_=nullptr;
        nt_gpu::gpu_free(dev_data_);
        dev_data_=nullptr;
    }

public:
    numTest() = default;

    numTest(size_t rows, size_t cols) :
        shape_(shape(rows, cols))
    {
        newArray();
    }

    numTest(const numpyArray& numpyInput)
    {
        const auto dataPtr = numpyInput.data();

        switch(numpyInput.ndim())
        {
            case 0:
            {
                break;
            }
            case 1:
            {
                const size_t size = static_cast<size_t>(numpyInput.size());
                shape_ = shape(1, size);
                newArray();
                std::copy(dataPtr, dataPtr + shape_.size(), begin());
                break;
            }
            case 2:
            {
                const size_t rows = static_cast<size_t>(numpyInput.shape(0));
                const size_t cols = static_cast<size_t>(numpyInput.shape(1));
                shape_ = shape(rows, cols);
                newArray();
                std::copy(dataPtr, dataPtr + shape_.size(), begin());
            }
            default:
            {
                break;
            }
        }
    }

    ~numTest()
    {
        deleteArray();
    }

public:
    dtype* begin()
    {
        return data_;
    }
    
    dtype* end()
    {
        return begin() + shape_.size();
    }

    dtype* cbegin(size_t row)
    {
        return begin() + (row * shape_.cols);
    }

    dtype* cend(size_t row)
    {
        return cbegin(row) + shape_.cols;
    }

    numTest<dtype>& fill(dtype value)
    {
        std::fill(begin(), end(), value);
        return *this;
    }

    dtype& operator()(size_t row, size_t col) noexcept
    {
        return data_[row * shape_.cols + col];
    }

    const dtype& operator()(size_t row, size_t col) const noexcept
    {
        return data_[row * shape_.cols + col];
    }

    void zeros()
    {
        this->fill(dtype{0});
    }

    void ones()
    {
        this->fill(dtype{1});
    }

    numpyArrayGeneric numpy()
    {
        const std::vector<pybind11::ssize_t> numpy_shape{static_cast<pybind11::ssize_t>(shape_.rows), 
            static_cast<pybind11::ssize_t>(shape_.cols)};
        const std::vector<pybind11::ssize_t> numpy_strides{static_cast<pybind11::ssize_t>(shape_.cols * sizeof(dtype)),
            static_cast<pybind11::ssize_t>(sizeof(dtype))};
        return numpyArrayGeneric(numpy_shape, numpy_strides, begin());
    }

    std::string str()
    {
        std::string out;
        out += "print start\n";
        out += "[";
        for (size_t row = 0; row < shape_.rows; ++row)
        {
            out += "[";
            for (size_t col = 0; col < shape_.cols; ++col)
            {
                out += std::to_string(operator()(row, col)) + ", ";
            }
            if (row == shape_.rows - 1)
            {
                out += "]";
            }
            else
            {
                out += "]\n";
            }
        }
        out += "]\n";
        out += "print end\n";
        return out;
    }

    void print_pointer(std::string str)
    {
        PRINT_DEBUG("call by %s this %p data %p\n", str.c_str(), this, data_);
        PRINT_DEBUG("call by %s this %p dev_data %p\n", str.c_str(), this, dev_data_);
    }

    void print()
    {
        if (is_cuda_)
        {
            this->cpu();
            std::cout << str();
            this->cuda();
        }
        else
        {
            std::cout << str();
        }
    }

    void cuda()
    {
        if (dev_data_ == nullptr) 
        {
            dev_data_ = nt_gpu::gpu_malloc(shape_.size() * sizeof(double));
            nt_gpu::copy_cpu_to_gpu(shape_.size() * sizeof(double), dev_data_, data_);
            nt_gpu::cpu_free(data_);

            print_pointer("cuda()");
            
            data_=nullptr;
            is_cuda_ = true;
        }
    }

    void cpu()
    {
        if (dev_data_)
        {
            data_ = (dtype*)nt_gpu::cpu_malloc(shape_.size() * sizeof(dtype));
            nt_gpu::copy_gpu_to_cpu(shape_.size() * sizeof(double), data_, dev_data_);
            nt_gpu::gpu_free(dev_data_);

            print_pointer("cpu()");

            dev_data_ = nullptr;
            is_cuda_ = false;
        }
    }

    void test()
    {
        printf("NumTest test\n");
    }
};

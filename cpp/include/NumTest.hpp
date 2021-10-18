#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <cuda.h>
#include "NumTest_gpu.hpp"

template<typename dtype>
using numpyArray = pybind11::array_t<dtype, pybind11::array::c_style>;
using numpyArrayGeneric = pybind11::array;

template<typename dtype>
class numTest
{
private:
    class shape
    {
    public:
        size_t rows{0};
        size_t cols{0};
        
        size_t size()
        {
            return rows * cols;
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

private:
    void newArray()
    {
        deleteArray();
        data_ = (dtype*)malloc(sizeof(dtype) * shape_.size());
        fill(dtype{0});
    }

    void newArray(size_t rows, size_t cols)
    {
        shape_ = shape(rows, cols);
        newArray();
    }

    void deleteArray()
    {
        nt_gpu::cpu_free(data_);
        data_=nullptr;
        nt_gpu::gpu_free(dev_data_);
        dev_data_=nullptr;
        // if (data_)
        // {
        //     free(data_);
        //     data_=nullptr;
        // }
    }

public:
    numTest() = default;

    numTest(size_t rows, size_t cols) :
        shape_(shape(rows, cols))
    {
        newArray();
    }

    numTest(const numpyArray<dtype>& numpyInput)
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

    numTest<dtype>& operator=(const numTest<dtype>& rhs) noexcept
    {
        if (rhs.size_ > 0)
        {
            newArray(shape_.rows, shape_.cols);
            std::copy(rhs.begin(), rhs.end(), begin());
        }
        return *this;
    }

    numpyArrayGeneric numpy()
    {
        const std::vector<pybind11::ssize_t> numpy_shape{static_cast<pybind11::ssize_t>(shape_.rows), 
            static_cast<pybind11::ssize_t>(shape_.cols)};
        const std::vector<pybind11::ssize_t> numpy_strides{static_cast<pybind11::ssize_t>(shape_.cols * sizeof(dtype)),
            static_cast<pybind11::ssize_t>(sizeof(dtype))};
        return numpyArrayGeneric(numpy_shape, numpy_strides, begin());
    }

    std::string str() const
    {
        std::string out;
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
        return out;
    }

    void print() const
    {
        std::cout << str();
    }

    void cuda()
    {
        if (dev_data_ == nullptr) 
        {
            dev_data_ = nt_gpu::gpu_malloc(shape_.size() * sizeof(double));
            nt_gpu::copy_cpu_to_gpu(shape_.size() * sizeof(double), dev_data_, data_);
            nt_gpu::cpu_free(data_);
            data_=nullptr;
        }
    }

    void cpu()
    {
        if (dev_data_)
        {
            data_ = (dtype*)nt_gpu::cpu_malloc(shape_.size() * sizeof(dtype));
            nt_gpu::copy_gpu_to_cpu(shape_.size() * sizeof(double), data_, dev_data_);
            nt_gpu::gpu_free(dev_data_);
            dev_data_ = nullptr;
        }
    }

    void gpu_mul()
    {
        nt_gpu::test_matrix_mul();
    }
};

namespace numTest_Functions
{
    template<typename dtype>
    void transpose(numTest<dtype>& returnArray, const numTest<dtype>& otherArray)
    {
        for (size_t row = 0; row < otherArray.shape_.rows; ++row)
        {
            for (size_t col = 0; col < otherArray.shape_.cols; ++col)
            {
                returnArray(col, row) = otherArray(row, col);
            }
        }
    }

    template<typename dtype>
    void dot_cpu(numTest<dtype>& returnArray, const numTest<dtype>& lhs, const numTest<dtype>& rhs)
    {
        if (!returnArray.data_)
        {
            printf("void dot_cpu(numTest<dtype>& returnArray, const numTest<dtype>& lhs, const numTest<dtype>& rhs) exception\n");   
            printf("returnArray data == nullptr\n");
            return;
        }

        if (lhs.shape_.cols != rhs.shape_.rows)
        {
            printf("void dot_cpu(numTest<dtype>& returnArray, const numTest<dtype>& lhs, const numTest<dtype>& rhs) exception\n");   
            printf("no match lhs, rhs shape\n");
            return;
        }

        for (size_t i = 0; i < lhs.shape_.rows; ++i)
        {
            for (size_t j = 0; j < rhs.shape_.cols; ++j)
            {
                dtype sum = dtype{0.0};

                for (size_t k = 0; k < lhs.shape_.cols; ++k)
                {
                    sum += lhs(i, k) * rhs(k, j);
                }

                returnArray(i, j) = sum;
            }
        }
    }

    template<typename dtype> 
    void dot_gpu(numTest<dtype>& returnArray, const numTest<dtype>& lhs, const numTest<dtype>& rhs)
    {
        if (returnArray.dev_data_)
        {
            nt_gpu::gpu_matrix_mul_double(returnArray.dev_data_, lhs.dev_data_, rhs.dev_data_,
                lhs.shape_.rows, lhs.shape_.cols, rhs.shape_.rows, rhs.shape_.cols);
        }
    }
}
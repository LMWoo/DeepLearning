#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <cuda.h>
#include "NumTest_gpu.hpp"
#include "NumTest.hpp"

namespace numTest_Functions
{
    void exception_print(std::string function_name, std::string exception_str)
    {
        printf("exception call by %s\n", function_name.c_str());   
        printf("%s\n", exception_str.c_str());
    }

    void null_check(std::string function_name, std::string pointer_name, void* ptr)
    {
        if (ptr == nullptr)
        {
            exception_print(function_name, pointer_name + " == nullptr");
        }
    }

    void test_dot_gpu()
    {
        nt_gpu::test_dot_gpu();
    }

    template<typename dtype>
    void copy_gpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void copy_gpu(numTest<dtype>*, const numTest<dtype>&)";
        null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        null_check(function_name, "otherArray.dev_data_", otherArray.dev_data_);

        nt_gpu::copy_gpu_to_gpu(otherArray.shape_.size() * sizeof(dtype), returnArray->dev_data_, otherArray.dev_data_);
    }

    template<typename dtype>
    void copy_cpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void copy_cpu(numTest<dtype>*, const numTest<dtype>&)";
        null_check(function_name, "returnArray->data_", returnArray->data_);
        null_check(function_name, "otherArray.data_", otherArray.data_);

        std::copy(otherArray.data_, otherArray.data_ + otherArray.shape_.size(), returnArray->data_);
    }

    template<typename dtype>
    void transpose_gpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void transpose_gpu(numTest<dtype>*, const numTest<dtype>&)";
        null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        null_check(function_name, "otherArray.dev_data_", otherArray.dev_data_);

        if (returnArray->shape_.cols != otherArray.shape_.rows || returnArray->shape_.rows != otherArray.shape_.cols)
        {
            exception_print(function_name, "no match returnArray, otherArray shape");
            return;
        }

        nt_gpu::transpose_gpu(returnArray->dev_data_, otherArray.dev_data_, otherArray.shape_.rows, otherArray.shape_.cols);  
    }

    template<typename dtype>
    void transpose_cpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void transpose_cpu(numTest<dtype>*, const numTest<dtype>&)";
        null_check(function_name, "returnArray->data_", returnArray->data_);
        null_check(function_name, "otherArray.data_", otherArray.data_);

        if (returnArray->shape_.cols != otherArray.shape_.rows || returnArray->shape_.rows != otherArray.shape_.cols)
        {
            exception_print(function_name, "no match returnArray, otherArray shape");
            return;
        }

        for (size_t row = 0; row < otherArray.shape_.rows; ++row)
        {
            for (size_t col = 0; col < otherArray.shape_.cols; ++col)
            {
                (*returnArray)(col, row) = otherArray(row, col);
            }
        }
    }


    template<typename dtype> 
    void dot_gpu(numTest<dtype>* returnArray, const numTest<dtype>& lhs, const numTest<dtype>& rhs)
    {
        std::string function_name = "void dot_gpu(numTest<dtype>*, const numTest<dtype>&, const numTest<dtype>&)";
        null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        null_check(function_name, "lhs.dev_data_", lhs.dev_data_);
        null_check(function_name, "rhs.dev_data_", rhs.dev_data_);

        if (!returnArray->dev_data_)
        {
            exception_print(function_name, "returnArray dev_data_ == nullptr");   
            return;
        }

        if (lhs.shape_.cols != rhs.shape_.rows)
        {
            exception_print(function_name, "no match lhs, rhs shape");
            return;
        }

        nt_gpu::matrix_dot_gpu(returnArray->dev_data_, lhs.dev_data_, rhs.dev_data_,
            lhs.shape_.rows, lhs.shape_.cols, rhs.shape_.rows, rhs.shape_.cols);
    }

    template<typename dtype>
    void dot_cpu(numTest<dtype>* returnArray, const numTest<dtype>& lhs, const numTest<dtype>& rhs)
    {
        std::string function_name = "void dot_cpu(numTest<dtype>&, const numTest<dtype>&, const numTest<dtype>&)";
        null_check(function_name, "returnArray->data_", returnArray->data_);
        null_check(function_name, "lhs.data_", lhs.data_);
        null_check(function_name, "rhs.data_", rhs.data_);

        if (lhs.shape_.cols != rhs.shape_.rows)
        {
            exception_print(function_name, "no match lhs, rhs shape");   
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

                (*returnArray)(i, j) = sum;
            }
        }
    }

    template<typename dtype>
    void add_gpu(numTest<dtype>* returnArray, const numTest<dtype>& lhs, const numTest<dtype>& rhs)
    {
        std::string function_name = "void add_gpu(numTest<dtype>*, const numTest<dtype>&, const numTest<dtype>&)";
        null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        null_check(function_name, "lhs.dev_data_", lhs.dev_data_);
        null_check(function_name, "rhs.dev_data_", rhs.dev_data_);
        
        if (lhs.shape_.rows != rhs.shape_.rows || lhs.shape_.cols != rhs.shape_.cols)
        {
            exception_print(function_name, "no no match lhs, rhs shape");
            return;
        }

        nt_gpu::add_gpu(returnArray->dev_data_, lhs.dev_data_, rhs.dev_data_, returnArray->shape_.rows, returnArray->shape_.cols);
    }

    template<typename dtype>
    void add_cpu(numTest<dtype>* returnArray, const numTest<dtype>& lhs, const numTest<dtype>& rhs)
    {
        std::string function_name = "void add_cpu(numTest<dtype>*, const numTest<dtype>&, const numTest<dtype>&)";
        null_check(function_name, "returnArray->data_", returnArray->data_);
        null_check(function_name, "lhs.data_", lhs.data_);
        null_check(function_name, "rhs.data_", rhs.data_);
        
        if (lhs.shape_.rows != rhs.shape_.rows || lhs.shape_.cols != rhs.shape_.cols)
        {
            exception_print(function_name, "no no match lhs, rhs shape");
            return;
        }

        std::transform(lhs.data_, lhs.data_ + lhs.shape_.size(), rhs.data_, returnArray->data_, std::plus<dtype>());
    }
}
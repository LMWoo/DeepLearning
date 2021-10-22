#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <cuda.h>
#include "NumTest_gpu.hpp"
#include "NumTest_Utils.hpp"
#include "NumTest.hpp"

namespace numTest_Functions
{
    void test_dot_gpu()
    {
        NumTest_gpu::test_dot_gpu();
    }

    template<typename dtype>
    void copy_gpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void copy_gpu(numTest<dtype>*, const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        NumTest_Utils::null_check(function_name, "otherArray.dev_data_", otherArray.dev_data_);

        NumTest_gpu::copy_gpu_to_gpu(otherArray.shape_.size() * sizeof(dtype), returnArray->dev_data_, otherArray.dev_data_);
    }

    template<typename dtype>
    void copy_cpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void copy_cpu(numTest<dtype>*, const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        NumTest_Utils::null_check(function_name, "otherArray.data_", otherArray.data_);

        std::copy(otherArray.data_, otherArray.data_ + otherArray.shape_.size(), returnArray->data_);
    }

    template<typename dtype>
    void transpose_gpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void transpose_gpu(numTest<dtype>*, const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        NumTest_Utils::null_check(function_name, "otherArray.dev_data_", otherArray.dev_data_);

        if (returnArray->shape_.cols != otherArray.shape_.rows || returnArray->shape_.rows != otherArray.shape_.cols)
        {
            NumTest_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
            return;
        }

        NumTest_gpu::transpose_gpu(returnArray->dev_data_, otherArray.dev_data_, otherArray.shape_.rows, otherArray.shape_.cols);  
    }

    template<typename dtype>
    void transpose_cpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void transpose_cpu(numTest<dtype>*, const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        NumTest_Utils::null_check(function_name, "otherArray.data_", otherArray.data_);

        if (returnArray->shape_.cols != otherArray.shape_.rows || returnArray->shape_.rows != otherArray.shape_.cols)
        {
            NumTest_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
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
        NumTest_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        NumTest_Utils::null_check(function_name, "lhs.dev_data_", lhs.dev_data_);
        NumTest_Utils::null_check(function_name, "rhs.dev_data_", rhs.dev_data_);

        if (lhs.shape_.cols != rhs.shape_.rows)
        {
            NumTest_Utils::exception_print(function_name, "no match lhs, rhs shape");
            return;
        }

        NumTest_gpu::matrix_dot_gpu(returnArray->dev_data_, lhs.dev_data_, rhs.dev_data_,
            lhs.shape_.rows, lhs.shape_.cols, rhs.shape_.rows, rhs.shape_.cols);
    }

    template<typename dtype>
    void dot_cpu(numTest<dtype>* returnArray, const numTest<dtype>& lhs, const numTest<dtype>& rhs)
    {
        std::string function_name = "void dot_cpu(numTest<dtype>&, const numTest<dtype>&, const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        NumTest_Utils::null_check(function_name, "lhs.data_", lhs.data_);
        NumTest_Utils::null_check(function_name, "rhs.data_", rhs.data_);

        if (lhs.shape_.cols != rhs.shape_.rows)
        {
            NumTest_Utils::exception_print(function_name, "no match lhs, rhs shape");   
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
        NumTest_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        NumTest_Utils::null_check(function_name, "lhs.dev_data_", lhs.dev_data_);
        NumTest_Utils::null_check(function_name, "rhs.dev_data_", rhs.dev_data_);
        
        if (lhs.shape_.rows != rhs.shape_.rows || lhs.shape_.cols != rhs.shape_.cols)
        {
            NumTest_Utils::exception_print(function_name, "no match lhs, rhs shape");
            return;
        }

        NumTest_gpu::add_gpu(returnArray->dev_data_, lhs.dev_data_, rhs.dev_data_, returnArray->shape_.rows, returnArray->shape_.cols);
    }

    template<typename dtype>
    void add_cpu(numTest<dtype>* returnArray, const numTest<dtype>& lhs, const numTest<dtype>& rhs)
    {
        std::string function_name = "void add_cpu(numTest<dtype>*, const numTest<dtype>&, const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        NumTest_Utils::null_check(function_name, "lhs.data_", lhs.data_);
        NumTest_Utils::null_check(function_name, "rhs.data_", rhs.data_);
        
        if (lhs.shape_.rows != rhs.shape_.rows || lhs.shape_.cols != rhs.shape_.cols)
        {
            NumTest_Utils::exception_print(function_name, "no match lhs, rhs shape");
            return;
        }

        std::transform(lhs.data_, lhs.data_ + lhs.shape_.size(), rhs.data_, returnArray->data_, std::plus<dtype>());
    }

    template<typename dtype>
    void tanh_gpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void tanh_gpu(numTest<dtype>*, const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        NumTest_Utils::null_check(function_name, "otherArray.dev_data_", otherArray.dev_data_);

        if (returnArray->shape_.rows != otherArray.shape_.rows || returnArray->shape_.cols != otherArray.shape_.cols)
        {
            NumTest_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
        }

        NumTest_gpu::tanh_gpu(returnArray->dev_data_, otherArray.dev_data_, returnArray->shape_.rows, returnArray->shape_.cols);
    }

    template<typename dtype>
    void tanh_cpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void tanh_cpu(numTest<dtype>*, const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        NumTest_Utils::null_check(function_name, "otherArray.data_", otherArray.data_);

        if (returnArray->shape_.rows != otherArray.shape_.rows || returnArray->shape_.cols != otherArray.shape_.cols)
        {
            NumTest_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
        }

        std::transform(otherArray.data_, otherArray.data_ + otherArray.shape_.size(), returnArray->data_,
            [](dtype inValue) -> auto
            {
                return std::tanh(inValue);
            });
    }

    template<typename dtype>
    void exp_gpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void exp_gpu(numTest<dtype>*, const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        NumTest_Utils::null_check(function_name, "otherArray.dev_data_", otherArray.dev_data_);

        if (returnArray->shape_.rows != otherArray.shape_.rows || returnArray->shape_.cols != otherArray.shape_.cols)
        {
            NumTest_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
        }

        NumTest_gpu::exp_gpu(returnArray->dev_data_, otherArray.dev_data_, returnArray->shape_.rows, returnArray->shape_.cols);
    }

    template<typename dtype>
    void exp_cpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void exp_cpu(numTest<dtype>*, const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        NumTest_Utils::null_check(function_name, "otherArray.data_", otherArray.data_);

        if (returnArray->shape_.rows != otherArray.shape_.rows || returnArray->shape_.cols != otherArray.shape_.cols)
        {
            NumTest_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
        }

        std::transform(otherArray.data_, otherArray.data_ + otherArray.shape_.size(), returnArray->data_,
            [](dtype inValue) -> auto
            {
                return std::exp(inValue);
            });
    }

    template<typename dtype>
    void sum_div_gpu(const numTest<dtype>& inArray)
    {
        std::string function_name = "void sum_div_gpu(const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "inArray.dev_data_", inArray.dev_data_);

        NumTest_gpu::sum_div_gpu(inArray.dev_data_, inArray.shape_.size());
    }

    template<typename dtype>
    dtype sum_cpu(const numTest<dtype>& inArray)
    {
        std::string function_name = "dtype sum_cpu(const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "inArray.data_", inArray.data_);

        return std::accumulate(inArray.data_, inArray.data_ + inArray.shape_.size(), dtype{0});
    }

    // template<typename dtype>
    // void div_gpu(numTest<dtype>* returnArray, double div)
    // {
    //     std::string function_name = "void div_gpu(numTest<dtype>*, dtype)";
    //     NumTest_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);

    //     NumTest_gpu::div_gpu(returnArray->dev_data_, div, returnArray->shape_.size());
    // }

    template<typename dtype>
    void div_cpu(numTest<dtype>* returnArray, dtype div)
    {
        std::string function_name = "void div_cpu(numTest<dtype>*, dtype)";
        NumTest_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);

        for (size_t i = 0; i < returnArray->shape_.size(); ++i)
        {
            returnArray->data_[i] /= (div + NumTest_Utils::epsilon<dtype>());
        }
        // std::transform(returnArray->data_, returnArray->data_ + returnArray->shape_.size(), returnArray->data_,
        //     [](dtype inValue) -> auto
        //     {
        //         return inValue / (div + NumTest_Utils::epsilon<dtype>());
        //     });
    }

    template<typename dtype>
    void softmax_gpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void softmax_gpu(numTest<dtype>*, const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        NumTest_Utils::null_check(function_name, "otherArray.dev_data_", otherArray.dev_data_);

        if (returnArray->shape_.rows != otherArray.shape_.rows || returnArray->shape_.cols != otherArray.shape_.cols)
        {
            NumTest_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
        }

        exp_gpu(returnArray, otherArray);
        sum_div_gpu(*returnArray);
        printf("softmax_gpu\n");
        returnArray->print();
    }

    template<typename dtype>
    void softmax_cpu(numTest<dtype>* returnArray, const numTest<dtype>& otherArray)
    {
        std::string function_name = "void softmax_cpu(numTest<dtype>*, const numTest<dtype>&)";
        NumTest_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        NumTest_Utils::null_check(function_name, "otherArray.data_", otherArray.data_);

        if (returnArray->shape_.rows != otherArray.shape_.rows || returnArray->shape_.cols != otherArray.shape_.cols)
        {
            NumTest_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
        }

        exp_cpu(returnArray, otherArray);
        dtype sum = sum_cpu(*returnArray);
        div_cpu(returnArray, sum);

        printf("softmax_cpu\n");
        returnArray->print();
    }
}
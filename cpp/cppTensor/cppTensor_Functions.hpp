#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <cuda.h>
#include "cppTensor_gpu.hpp"
#include "cppTensor_Utils.hpp"
#include "cppTensor.hpp"

namespace cppTensor_Functions
{
    void test_matMul_gpu()
    {
        cppTensor_gpu::test_matMul_gpu();
    }

    template<typename dtype>
    void copy_gpu(cppTensor<dtype>* returnArray, const cppTensor<dtype>& otherArray)
    {
        std::string function_name = "void copy_gpu(cppTensor<dtype>*, const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        cppTensor_Utils::null_check(function_name, "otherArray.dev_data_", otherArray.dev_data_);

        cppTensor_gpu::copy_gpu_to_gpu(otherArray.shape_.size() * sizeof(dtype), returnArray->dev_data_, otherArray.dev_data_);
    }

    template<typename dtype>
    void copy_cpu(cppTensor<dtype>* returnArray, const cppTensor<dtype>& otherArray)
    {
        std::string function_name = "void copy_cpu(cppTensor<dtype>*, const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        cppTensor_Utils::null_check(function_name, "otherArray.data_", otherArray.data_);

        std::copy(otherArray.data_, otherArray.data_ + otherArray.shape_.size(), returnArray->data_);
    }

    template<typename dtype>
    void transpose_gpu(cppTensor<dtype>& returnArray, const cppTensor<dtype>& otherArray)
    {
        std::string function_name = "void transpose_gpu(cppTensor<dtype>&, const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray.dev_data_", returnArray.dev_data_);
        cppTensor_Utils::null_check(function_name, "otherArray.dev_data_", otherArray.dev_data_);

        if (returnArray.shape_.cols != otherArray.shape_.rows || returnArray.shape_.rows != otherArray.shape_.cols)
        {
            cppTensor_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
            return;
        }

        cppTensor_gpu::transpose_gpu(returnArray.dev_data_, otherArray.dev_data_, otherArray.shape_.rows, otherArray.shape_.cols);  
    }

    template<typename dtype>
    void transpose_cpu(cppTensor<dtype>* returnArray, const cppTensor<dtype>& otherArray)
    {
        std::string function_name = "void transpose_cpu(cppTensor<dtype>*, const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        cppTensor_Utils::null_check(function_name, "otherArray.data_", otherArray.data_);

        if (returnArray->shape_.cols != otherArray.shape_.rows || returnArray->shape_.rows != otherArray.shape_.cols)
        {
            cppTensor_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
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
    void matMul_gpu(cppTensor<dtype>& returnArray, const cppTensor<dtype>& lhs, const cppTensor<dtype>& rhs, bool useSharedMemory)
    {
        std::string function_name = "void matMul_gpu(cppTensor<dtype>&, const cppTensor<dtype>&, const cppTensor<dtype>&, bool)";
        cppTensor_Utils::null_check(function_name, "returnArray.dev_data_", returnArray.dev_data_);
        cppTensor_Utils::null_check(function_name, "lhs.dev_data_", lhs.dev_data_);
        cppTensor_Utils::null_check(function_name, "rhs.dev_data_", rhs.dev_data_);

        if (lhs.shape_.cols != rhs.shape_.rows)
        {
            cppTensor_Utils::exception_print(function_name, "no match lhs, rhs shape");
            return;
        }

        cppTensor_gpu::matMul_gpu(returnArray.dev_data_, lhs.dev_data_, rhs.dev_data_,
            lhs.shape_.rows, lhs.shape_.cols, rhs.shape_.rows, rhs.shape_.cols, useSharedMemory);
    }

    template<typename dtype>
    void matMul_cpu(cppTensor<dtype>* returnArray, const cppTensor<dtype>& lhs, const cppTensor<dtype>& rhs)
    {
        std::string function_name = "void matMul_cpu(cppTensor<dtype>&, const cppTensor<dtype>&, const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        cppTensor_Utils::null_check(function_name, "lhs.data_", lhs.data_);
        cppTensor_Utils::null_check(function_name, "rhs.data_", rhs.data_);

        if (lhs.shape_.cols != rhs.shape_.rows)
        {
            cppTensor_Utils::exception_print(function_name, "no match lhs, rhs shape");   
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
    void add_gpu(cppTensor<dtype>& returnArray, const cppTensor<dtype>& lhs, const cppTensor<dtype>& rhs)
    {
        std::string function_name = "void add_gpu(cppTensor<dtype>&, const cppTensor<dtype>&, const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray.dev_data_", returnArray.dev_data_);
        cppTensor_Utils::null_check(function_name, "lhs.dev_data_", lhs.dev_data_);
        cppTensor_Utils::null_check(function_name, "rhs.dev_data_", rhs.dev_data_);
        
        if (lhs.shape_.rows != rhs.shape_.rows || lhs.shape_.cols != rhs.shape_.cols)
        {
            cppTensor_Utils::exception_print(function_name, "no match lhs, rhs shape");
            return;
        }

        cppTensor_gpu::add_gpu(returnArray.dev_data_, lhs.dev_data_, rhs.dev_data_, returnArray.shape_.size());
    }

    template<typename dtype>
    void add_cpu(cppTensor<dtype>* returnArray, const cppTensor<dtype>& lhs, const cppTensor<dtype>& rhs)
    {
        std::string function_name = "void add_cpu(cppTensor<dtype>*, const cppTensor<dtype>&, const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        cppTensor_Utils::null_check(function_name, "lhs.data_", lhs.data_);
        cppTensor_Utils::null_check(function_name, "rhs.data_", rhs.data_);
        
        if (lhs.shape_.rows != rhs.shape_.rows || lhs.shape_.cols != rhs.shape_.cols)
        {
            cppTensor_Utils::exception_print(function_name, "no match lhs, rhs shape");
            return;
        }

        std::transform(lhs.data_, lhs.data_ + lhs.shape_.size(), rhs.data_, returnArray->data_, std::plus<dtype>());
    }

    template<typename dtype>
    void tanh_gpu(cppTensor<dtype>* returnArray, const cppTensor<dtype>& otherArray)
    {
        std::string function_name = "void tanh_gpu(cppTensor<dtype>*, const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        cppTensor_Utils::null_check(function_name, "otherArray.dev_data_", otherArray.dev_data_);

        if (returnArray->shape_.rows != otherArray.shape_.rows || returnArray->shape_.cols != otherArray.shape_.cols)
        {
            cppTensor_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
        }

        cppTensor_gpu::tanh_gpu(returnArray->dev_data_, otherArray.dev_data_, returnArray->shape_.rows, returnArray->shape_.cols);
    }

    template<typename dtype>
    void tanh_cpu(cppTensor<dtype>* returnArray, const cppTensor<dtype>& otherArray)
    {
        std::string function_name = "void tanh_cpu(cppTensor<dtype>*, const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        cppTensor_Utils::null_check(function_name, "otherArray.data_", otherArray.data_);

        if (returnArray->shape_.rows != otherArray.shape_.rows || returnArray->shape_.cols != otherArray.shape_.cols)
        {
            cppTensor_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
        }

        std::transform(otherArray.data_, otherArray.data_ + otherArray.shape_.size(), returnArray->data_,
            [](dtype inValue) -> auto
            {
                return std::tanh(inValue);
            });
    }

    template<typename dtype>
    void exp_gpu(cppTensor<dtype>* returnArray, const cppTensor<dtype>& otherArray)
    {
        std::string function_name = "void exp_gpu(cppTensor<dtype>*, const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        cppTensor_Utils::null_check(function_name, "otherArray.dev_data_", otherArray.dev_data_);

        if (returnArray->shape_.rows != otherArray.shape_.rows || returnArray->shape_.cols != otherArray.shape_.cols)
        {
            cppTensor_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
        }

        cppTensor_gpu::exp_gpu(returnArray->dev_data_, otherArray.dev_data_, returnArray->shape_.rows, returnArray->shape_.cols);
    }

    template<typename dtype>
    void exp_cpu(cppTensor<dtype>* returnArray, const cppTensor<dtype>& otherArray)
    {
        std::string function_name = "void exp_cpu(cppTensor<dtype>*, const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        cppTensor_Utils::null_check(function_name, "otherArray.data_", otherArray.data_);

        if (returnArray->shape_.rows != otherArray.shape_.rows || returnArray->shape_.cols != otherArray.shape_.cols)
        {
            cppTensor_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
        }

        std::transform(otherArray.data_, otherArray.data_ + otherArray.shape_.size(), returnArray->data_,
            [](dtype inValue) -> auto
            {
                return std::exp(inValue);
            });
    }

    template<typename dtype>
    void sum_div_gpu(const cppTensor<dtype>& inArray)
    {
        std::string function_name = "void sum_div_gpu(const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "inArray.dev_data_", inArray.dev_data_);

        cppTensor_gpu::sum_div_gpu(inArray.dev_data_, inArray.shape_.size());
    }

    template<typename dtype>
    dtype sum_cpu(const cppTensor<dtype>& inArray)
    {
        std::string function_name = "dtype sum_cpu(const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "inArray.data_", inArray.data_);

        return std::accumulate(inArray.data_, inArray.data_ + inArray.shape_.size(), dtype{0});
    }

    // template<typename dtype>
    // void div_gpu(cppTensor<dtype>* returnArray, double div)
    // {
    //     std::string function_name = "void div_gpu(cppTensor<dtype>*, dtype)";
    //     cppTensor_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);

    //     cppTensor_gpu::div_gpu(returnArray->dev_data_, div, returnArray->shape_.size());
    // }

    template<typename dtype>
    void div_cpu(cppTensor<dtype>* returnArray, dtype div)
    {
        std::string function_name = "void div_cpu(cppTensor<dtype>*, dtype)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);

        for (size_t i = 0; i < returnArray->shape_.size(); ++i)
        {
            returnArray->data_[i] /= (div + cppTensor_Utils::epsilon<dtype>());
        }
        // std::transform(returnArray->data_, returnArray->data_ + returnArray->shape_.size(), returnArray->data_,
        //     [](dtype inValue) -> auto
        //     {
        //         return inValue / (div + cppTensor_Utils::epsilon<dtype>());
        //     });
    }

    template<typename dtype>
    void softmax_gpu(cppTensor<dtype>* returnArray, const cppTensor<dtype>& otherArray)
    {
        std::string function_name = "void softmax_gpu(cppTensor<dtype>*, const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        cppTensor_Utils::null_check(function_name, "otherArray.dev_data_", otherArray.dev_data_);

        if (returnArray->shape_.rows != otherArray.shape_.rows || returnArray->shape_.cols != otherArray.shape_.cols)
        {
            cppTensor_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
        }

        exp_gpu(returnArray, otherArray);
        sum_div_gpu(*returnArray);
    }

    template<typename dtype>
    void softmax_cpu(cppTensor<dtype>* returnArray, const cppTensor<dtype>& otherArray)
    {
        std::string function_name = "void softmax_cpu(cppTensor<dtype>*, const cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        cppTensor_Utils::null_check(function_name, "otherArray.data_", otherArray.data_);

        if (returnArray->shape_.rows != otherArray.shape_.rows || returnArray->shape_.cols != otherArray.shape_.cols)
        {
            cppTensor_Utils::exception_print(function_name, "no match returnArray, otherArray shape");
        }

        exp_cpu(returnArray, otherArray);
        dtype sum = sum_cpu(*returnArray);
        div_cpu(returnArray, sum);
    }

    template<typename dtype>
    void minus_gpu(cppTensor<dtype>* returnArray)
    {
        std::string function_name = "void minus_gpu(cppTensor<dtype>*)";
        cppTensor_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);

        cppTensor_gpu::minus_gpu(returnArray->dev_data_, returnArray->shape_.size());
    }

    template<typename dtype>
    void minus_cpu(cppTensor<dtype>* returnArray)
    {
        std::string function_name = "void minus_cpu(cppTensor<dtype>*)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);

        for (size_t i = 0; i < returnArray->shape_.size(); ++i)
        {
            returnArray->data_[i] = -returnArray->data_[i];
        }
    }

    template<typename dtype>
    void log_gpu(cppTensor<dtype>* returnArray)
    {
        std::string function_name = "void log_gpu(cppTensor<dtype>*)";
        cppTensor_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);

        cppTensor_gpu::log_gpu(returnArray->dev_data_, returnArray->shape_.size());
    }

    template<typename dtype>
    void log_cpu(cppTensor<dtype>* returnArray)
    {
        std::string function_name = "void log_cpu(cppTensor<dtype>*)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);

        for (size_t i = 0; i < returnArray->shape_.size(); ++i)
        {
            returnArray->data_[i] = std::log(returnArray->data_[i]);
        }
    }

    template<typename dtype>
    void deriv_softmax_gpu(cppTensor<dtype>& dY, cppTensor<dtype>& loss, cppTensor<dtype>& Y, const cppTensor<dtype>& labels)
    {
        std::string function_name = "void deriv_softmax_gpu(cppTensor<dtype>&, cppTensor<dtype>&, cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "loss.dev_data_", loss.dev_data_);
        cppTensor_Utils::null_check(function_name, "Y.dev_data_", Y.dev_data_);
        cppTensor_Utils::null_check(function_name, "labels.dev_data_", labels.dev_data_);

        cppTensor_gpu::deriv_softmax_gpu(loss.shape_.size(), dY.dev_data_, loss.dev_data_, Y.dev_data_, labels.dev_data_);
    }

    template<typename dtype>
    void deriv_softmax_cpu(cppTensor<dtype>& dY, cppTensor<dtype>& loss, cppTensor<dtype>& Y, const cppTensor<dtype>& labels)
    {
        std::string function_name = "void deriv_softmax_cpu(cppTensor<dtype>&, cppTensor<dtype>&, cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "loss.data_", loss.data_);
        cppTensor_Utils::null_check(function_name, "Y.data_", Y.data_);
        cppTensor_Utils::null_check(function_name, "labels.data_", labels.data_);

        int index = const_cast<cppTensor<dtype>&>(labels)(0, 0);
        dY(index, 0) -= 1;
        loss(index, 0) = Y(index, 0);
    }

    template<typename dtype>
    void mul_gpu(cppTensor<dtype>* returnArray, dtype inValue)
    {
        std::string function_name = "void mul_gpu(cppTensor<dtype>*, dtype)";
        cppTensor_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);

        cppTensor_gpu::mul_gpu(returnArray->dev_data_, inValue, returnArray->shape_.size());
    }

    template<typename dtype>
    void mul_cpu(cppTensor<dtype>* returnArray, dtype inValue)
    {
        std::string function_name = "void mul_cpu(cppTensor<dtype>*, dtype)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);

        for (size_t i = 0; i < returnArray->shape_.size(); ++i)
        {
            returnArray->data_[i] = returnArray->data_[i] * inValue;
        }
    }

    template<typename dtype>
    void mul_gpu(cppTensor<dtype>* returnArray, cppTensor<dtype>& lhs, cppTensor<dtype>& rhs)
    {
        std::string function_name = "void mul_gpu(cppTensor<dtype>*, cppTensor<dtype>&, cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        cppTensor_Utils::null_check(function_name, "lhs.dev_data_", lhs.dev_data_);
        cppTensor_Utils::null_check(function_name, "rhs.dev_data_", rhs.dev_data_);

        cppTensor_gpu::mul_gpu(returnArray->dev_data_, lhs.dev_data_, rhs.dev_data_, returnArray->shape_.size());
    }

    template<typename dtype>
    void mul_cpu(cppTensor<dtype>* returnArray, cppTensor<dtype>& lhs, cppTensor<dtype>& rhs)
    {
        std::string function_name = "void mul_cpu(cppTensor<dtype>*, cppTensor<dtype>&, cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        cppTensor_Utils::null_check(function_name, "lhs.data_", lhs.data_);
        cppTensor_Utils::null_check(function_name, "rhs.data_", rhs.data_);

        for (size_t i = 0; i < returnArray->shape_.size(); ++i)
        {
            returnArray->data_[i] = lhs.data_[i] * rhs.data_[i];
        }
    }


    template<typename dtype>
    void deriv_tanh_gpu(cppTensor<dtype>* returnArray, cppTensor<dtype>& otherArray)
    {
        std::string function_name = "void deriv_tanh_gpu(cppTensor<dtype>*, cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);
        cppTensor_Utils::null_check(function_name, "otherArray.dev_data_", otherArray.dev_data_);

        cppTensor_gpu::deriv_tanh_gpu(returnArray->dev_data_, otherArray.dev_data_, returnArray->shape_.size());
    }

    template<typename dtype>
    void deriv_tanh_cpu(cppTensor<dtype>* returnArray, cppTensor<dtype>& otherArray)
    {
        std::string function_name = "void deriv_tanh_cpu(cppTensor<dtype>*, cppTensor<dtype>&)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);
        cppTensor_Utils::null_check(function_name, "otherArray.data_", otherArray.data_);

        for (size_t i = 0; i < returnArray->shape_.size(); ++i)
        {
            returnArray->data_[i] = 1.0 - otherArray.data_[i] * otherArray.data_[i];
        }
    }

    template<typename dtype>
    void clip_gpu(cppTensor<dtype>* returnArray, const dtype& low, const dtype& high)
    {
        std::string function_name = "void clip_gpu(cppTensor<dtype>*, const dtype&, const dtype&)";
        cppTensor_Utils::null_check(function_name, "returnArray->dev_data_", returnArray->dev_data_);

        cppTensor_gpu::clip_gpu(returnArray->dev_data_, low, high, returnArray->shape_.size());
    }

    template<typename dtype>
    void clip_cpu(cppTensor<dtype>* returnArray, const dtype& low, const dtype& high)
    {
        std::string function_name = "void clip_cpu(cppTensor<dtype>*, const dtype&, const dtype&)";
        cppTensor_Utils::null_check(function_name, "returnArray->data_", returnArray->data_);

        for (size_t i = 0; i < returnArray->shape_.size(); ++i)
        {
            returnArray->data_[i] = std::max(low, std::min(high, returnArray->data_[i]));
        }
    }

    template<typename dtype>
    void optimizer_gpu(cppTensor<dtype>* param, cppTensor<dtype>* mem, const cppTensor<dtype>& dparam, double lr)
    {
        std::string function_name = "void optimizer_gpu(cppTensor<dtype>*, cppTensor<dtype>*, const cppTensor<dtype>&, double)";
        cppTensor_Utils::null_check(function_name, "param->dev_data_", param->dev_data_);
        cppTensor_Utils::null_check(function_name, "mem->dev_data_", mem->dev_data_);
        cppTensor_Utils::null_check(function_name, "dparam.dev_data_", dparam.dev_data_);

        cppTensor_gpu::optimizer_gpu(param->dev_data_, mem->dev_data_, dparam.dev_data_, param->shape_.size());
    }

    template<typename dtype>
    void optimizer_cpu(cppTensor<dtype>* param, cppTensor<dtype>* mem, const cppTensor<dtype>& dparam, double lr)
    {
        std::string function_name = "void optimizer_cpu(cppTensor<dtype>*, cppTensor<dtype>*, const cppTensor<dtype>&, double)";
        cppTensor_Utils::null_check(function_name, "param->data_", param->data_);
        cppTensor_Utils::null_check(function_name, "mem->data_", mem->data_);
        cppTensor_Utils::null_check(function_name, "dparam.data_", dparam.data_);

        for (size_t i = 0; i < param->shape_.size(); ++i)
        {
            mem->data_[i] += dparam.data_[i] * dparam.data_[i];
            param->data_[i] += (lr * dparam.data_[i]) / sqrt(mem->data_[i] + 1e-8);
        }
    }
}
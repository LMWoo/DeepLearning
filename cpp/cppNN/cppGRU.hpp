#pragma once

#include "cppNN.hpp"

template<typename dtype>
class cppGRU : public cppNN<dtype>
{
public:
    using numpyArray = pybind11::array_t<dtype, pybind11::array::c_style>;
    using numpyArrayGeneric = pybind11::array;
    using cppTensorType = cppTensor<dtype>;
    using mapIntCppTensor = std::unordered_map<int, cppTensor<dtype>*>;
    using mapStrCppTensor = std::unordered_map<std::string, cppTensor<dtype>*>;
    using mapIntCppTensorIter = std::unordered_map<int, cppTensor<double>*>::iterator;
    using mapStrCppTensorIter = std::unordered_map<std::string, cppTensor<double>*>::iterator;

    cppGRU()
    {
        printf("cppGRU()\n");
    }

    ~cppGRU()
    {
        printf("~cppGRU()\n");
    }
protected:
    virtual void cuda_impl() override
    {
        printf("gru cuda_impl()\n");
    }

    virtual void cpu_impl() override
    {
        printf("gru cpu_impl()\n");
    }

    virtual mapStrCppTensor parameters_impl() override
    {
        return mapStrCppTensor();
    }

    virtual void backward_impl(const cppTensor<dtype>& dY) override
    {
        printf("gru backward_impl\n");
    }

    virtual void cross_entropy_loss_impl(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) override
    {
        printf("gru cross_entropy_loss_impl\n");
    }

    virtual cppTensor<dtype> forward_impl(const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev) override
    {
        printf("gru forward_impl\n");
        return cppTensor<dtype>();
    }
};
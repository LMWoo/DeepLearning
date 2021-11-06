#pragma once

#include "cppNN.hpp"

template<typename dtype>
class cppLSTM : public cppNN<dtype>
{
public:
    using numpyArray = pybind11::array_t<dtype, pybind11::array::c_style>;
    using numpyArrayGeneric = pybind11::array;
    using cppTensorType = cppTensor<dtype>;
    using mapIntCppTensor = std::unordered_map<int, cppTensor<dtype>*>;
    using mapStrCppTensor = std::unordered_map<std::string, cppTensor<dtype>*>;
    using mapIntCppTensorIter = std::unordered_map<int, cppTensor<double>*>::iterator;
    using mapStrCppTensorIter = std::unordered_map<std::string, cppTensor<double>*>::iterator;

    cppLSTM()
    {
        printf("cppLSTM()\n");
    }

    ~cppLSTM()
    {
        printf("~cppLSTM()\n");
    }
protected:
    virtual void cuda_impl() override
    {
        printf("lstm cuda_impl()\n");
    }

    virtual void cpu_impl() override
    {
        printf("lstm cpu_impl()\n");
    }

    virtual mapStrCppTensor parameters_impl() override
    {
        return mapStrCppTensor();
    }

    virtual std::vector<cppTensor<dtype>> backward_impl(const cppTensor<dtype>& dY) override
    {
        printf("lstm backward_impl\n");
        return std::vector<cppTensor<dtype>>({cppTensor<dtype>()});
    }

    virtual void cross_entropy_loss_impl(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) override
    {
        printf("lstm cross_entropy_loss_impl\n");
    }

    virtual cppTensor<dtype> forward_impl(const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev) override
    {
        printf("lstm forward_impl\n");
        return cppTensor<dtype>();
    }
};
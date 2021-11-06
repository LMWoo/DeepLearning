#pragma once

#include "cppTensor/cppTensor.hpp"
#include "cppTensor/cppTensor_Utils.hpp"
#include "cppTensor/cppTensor_Functions.hpp"
#include <unordered_map>
#include <tuple>

using namespace cppTensor_Functions;

template<typename dtype>
class cppNN
{
public:
    using numpyArray = pybind11::array_t<dtype, pybind11::array::c_style>;
    using numpyArrayGeneric = pybind11::array;
    using cppTensorType = cppTensor<dtype>;
    using mapIntCppTensor = std::unordered_map<int, cppTensor<dtype>*>;
    using mapStrCppTensor = std::unordered_map<std::string, cppTensor<dtype>*>;
    using mapIntCppTensorIter = std::unordered_map<int, cppTensor<double>*>::iterator;
    using mapStrCppTensorIter = std::unordered_map<std::string, cppTensor<double>*>::iterator;

public:
    cppNN()
    {
        printf("cppNN()\n");
    }

    virtual ~cppNN()
    {
        printf("~cppNN()\n");
    }

    void cuda()
    {
        is_cuda_=true;

        cuda_impl();
    }

    void cpu()
    {
        is_cuda_=false;

        cpu_impl();
    }

    void useSharedMemory()
    {
        this->use_sharedMemory=true;
    }

    void notUseSharedMemory()
    {
        this->use_sharedMemory=false;
    }

public:
    cppTensor<dtype> forward(std::vector<cppTensor<dtype>>& x,  cppTensor<dtype>& hprev)
    {
        return forward_impl(x, hprev);
    }

    mapStrCppTensor parameters()
    {
        return parameters_impl();
    }

    void cross_entropy_loss(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels)
    {
        cross_entropy_loss_impl(dY, Y, loss, outputs, labels);
    }

    void backward(cppTensor<dtype>& dY)
    {
        backward_impl(dY);
    }

protected:
    virtual void cuda_impl() = 0;

    virtual void cpu_impl() = 0;

    virtual mapStrCppTensor parameters_impl() = 0;

    virtual void backward_impl(const cppTensor<dtype>& dY) = 0;

    virtual void cross_entropy_loss_impl(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) = 0;

    virtual cppTensor<dtype> forward_impl(const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev) = 0;

protected:
    bool is_cuda_{false};
    bool use_sharedMemory{false};
};
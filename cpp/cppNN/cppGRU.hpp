#pragma once

#include "cppNN.hpp"

template<typename dtype>
class cppGRU : public cppNN<dtype>
{
public:
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

    virtual void optimizer_impl() override
    {
        printf("gru optimizer_impl\n");
    }

    virtual std::vector<cppTensor<dtype>> backward_impl(const cppTensor<dtype>& dY) override
    {
        printf("gru backward_impl\n");
        return std::vector<cppTensor<dtype>>({cppTensor<dtype>()});
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
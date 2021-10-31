#pragma once

#include "cppNN.hpp"

template<typename dtype>
class cppLSTM : public cppNN<dtype>
{
public:
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

    virtual void optimizer_impl() override
    {
        printf("lstm optimizer_impl\n");
    }

    virtual void backward_impl(const cppTensor<dtype>& dY) override
    {
        printf("lstm backward_impl\n");
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
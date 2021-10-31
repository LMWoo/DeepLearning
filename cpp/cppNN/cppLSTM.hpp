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
    virtual void cuda_child() override
    {
        printf("lstm cuda_child()\n");
    }

    virtual void cpu_child() override
    {
        printf("lstm cpu_child()\n");
    }

    virtual void optimizer_gpu() override
    {
        printf("lstm optimizer_gpu\n");
    }

    virtual void optimizer_cpu() override
    {
        printf("lstm optimizer_cpu\n");
    }

    virtual void backward_gpu(cppTensor<dtype>& dY) override
    {
        printf("lstm backward_gpu\n");
    }
    
    virtual void backward_cpu(cppTensor<dtype>& dY) override
    {
        printf("lstm backward_cpu\n");
    }

    virtual void cross_entropy_loss_gpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) override
    {
        printf("lstm cross_entropy_loss_gpu\n");
    }

    virtual void cross_entropy_loss_cpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) override
    {
        printf("lstm cross_entropy_loss_cpu\n");
    }

    virtual cppTensor<dtype> forward_impl(const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev) override
    {
        printf("lstm forward_impl\n");
        return cppTensor<dtype>();
    }
};
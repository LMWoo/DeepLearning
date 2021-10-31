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
    virtual void cuda_child() override
    {
        printf("gru cuda_child()\n");
    }

    virtual void cpu_child() override
    {
        printf("gru cpu_child()\n");
    }

    virtual void optimizer_gpu() override
    {
        printf("gru optimizer_gpu\n");
    }

    virtual void optimizer_cpu() override
    {
        printf("gru optimizer_cpu\n");
    }

    virtual void backward_gpu(cppTensor<dtype>& dY) override
    {
        printf("gru backward_gpu\n");
    }
    
    virtual void backward_cpu(cppTensor<dtype>& dY) override
    {
        printf("gru backward_cpu\n");
    }

    virtual void cross_entropy_loss_gpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) override
    {
        printf("gru cross_entropy_loss_gpu\n");
    }

    virtual void cross_entropy_loss_cpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) override
    {
        printf("gru cross_entropy_loss_cpu\n");
    }

    virtual cppTensor<dtype> forward_impl(const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev) override
    {
        printf("gru forward_impl\n");
        return cppTensor<dtype>();
    }
};
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
    virtual void optimizer_gpu()
    {
        printf("gru optimizer_gpu\n");
    }

    virtual void optimizer_cpu()
    {
        printf("gru optimizer_cpu\n");
    }

    virtual void backward_gpu(cppTensor<dtype>& dY)
    {
        printf("gru backward_gpu\n");
    }
    
    virtual void backward_cpu(cppTensor<dtype>& dY)
    {
        printf("gru backward_cpu\n");
    }

    virtual void cross_entropy_loss_gpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels)
    {
        printf("gru cross_entropy_loss_gpu\n");
    }

    virtual void cross_entropy_loss_cpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels)
    {
        printf("gru cross_entropy_loss_cpu\n");
    }

    virtual void forward_gpu(cppTensor<dtype>& outputs, const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev)
    {
        printf("gru forward_gpu\n");
    }

    virtual void forward_cpu(cppTensor<dtype>& outputs, const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev)
    {
        printf("gru forward_cpu\n");
    }
};
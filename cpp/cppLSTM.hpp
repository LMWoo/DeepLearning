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
    virtual void optimizer_gpu()
    {
        printf("lstm optimizer_gpu\n");
    }

    virtual void optimizer_cpu()
    {
        printf("lstm optimizer_cpu\n");
    }

    virtual void backward_gpu(cppTensor<dtype>& dY)
    {
        printf("lstm backward_gpu\n");
    }
    
    virtual void backward_cpu(cppTensor<dtype>& dY)
    {
        printf("lstm backward_cpu\n");
    }

    virtual void cross_entropy_loss_gpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels)
    {
        printf("lstm cross_entropy_loss_gpu\n");
    }

    virtual void cross_entropy_loss_cpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels)
    {
        printf("lstm cross_entropy_loss_cpu\n");
    }

    virtual void forward_gpu(cppTensor<dtype>& outputs, const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev)
    {
        printf("lstm forward_gpu\n");
    }

    virtual void forward_cpu(cppTensor<dtype>& outputs, const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev)
    {
        printf("lstm forward_cpu\n");
    }
};
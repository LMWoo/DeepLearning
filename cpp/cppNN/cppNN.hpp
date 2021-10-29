#pragma once

#include "cppTensor/cppTensor.hpp"
#include "cppTensor/cppTensor_Utils.hpp"
#include "cppTensor/cppTensor_Functions.hpp"
#include <unordered_map>
#include <tuple>

template<typename dtype>
class cppNN
{
public:
    using numpyArray = pybind11::array_t<dtype, pybind11::array::c_style>;
    using numpyArrayGeneric = pybind11::array;
    using cppTensorType = cppTensor<dtype>;
    using cppTensorTypeMap = std::unordered_map<int, cppTensor<dtype>*>;
    using cppTensorTypeMapDoubleIter = std::unordered_map<int, cppTensor<double>*>::iterator;

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

        cuda_child();
    }

    void cpu()
    {
        is_cuda_=false;

        cpu_child();
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
    void forward(cppTensor<dtype>& outputs,  std::vector<cppTensor<dtype>>& x,  cppTensor<dtype>& hprev)
    {
        if (this->is_cuda_)
        {
            forward_gpu(outputs, x, hprev);
        }
        else
        {
            forward_cpu(outputs, x, hprev);
        }

        // outputs.data_=nullptr;
        // hprev.data_=nullptr;
        
        for (int i = 0; i < x.size(); ++i)
        {
            // x[i].print();
            x[i].data_=nullptr;
            x[i].dev_data_=nullptr;
        }
    }


    void cross_entropy_loss(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels)
    {
        if (this->is_cuda_)
        {
            cross_entropy_loss_gpu(dY, Y, loss, outputs, labels);
        }
        else
        {
            cross_entropy_loss_cpu(dY, Y, loss, outputs, labels);
        }
    }

    void backward(cppTensor<dtype>& dY)
    {
        if (this->is_cuda_)
        {
            backward_gpu(dY);
        }
        else
        {
            backward_cpu(dY);
        }
    }

    void optimizer()
    {
        if (this->is_cuda_)
        {
            optimizer_gpu();
        }
        else
        {
            optimizer_cpu();
        }
    }
    
protected:
    virtual void cuda_child() = 0;

    virtual void cpu_child() = 0;

    virtual void optimizer_gpu() = 0;

    virtual void optimizer_cpu() = 0;

    virtual void backward_gpu(cppTensor<dtype>& dY) = 0;
    
    virtual void backward_cpu(cppTensor<dtype>& dY) = 0;

    virtual void cross_entropy_loss_gpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) = 0;

    virtual void cross_entropy_loss_cpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) = 0;

    virtual void forward_gpu(cppTensor<dtype>& outputs, const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev) = 0;

    virtual void forward_cpu(cppTensor<dtype>& outputs, const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev) = 0;

protected:
    bool is_cuda_{false};
    bool use_sharedMemory{false};
};
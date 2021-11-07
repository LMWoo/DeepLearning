#pragma once

#include "cppLoss.hpp"

template<typename dtype>
class cppCrossEntropyLoss : public cppLoss<dtype>
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
    cppCrossEntropyLoss()
    {
        
    }

    ~cppCrossEntropyLoss()
    {
        SAFE_DELETE(this->loss)
        SAFE_DELETE(this->Y)
        SAFE_DELETE(this->dY)
    }

protected:
    virtual const cppTensor<dtype>& operator_impl(const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) override
    {
        if (this->loss == nullptr && this->Y == nullptr && this->dY == nullptr)
        {
            if (outputs.is_cuda_ && labels.is_cuda_)
            {
                this->loss = new cppTensor<dtype>(outputs.shape_.rows, outputs.shape_.cols, true);
                this->Y = new cppTensor<dtype>(outputs.shape_.rows, outputs.shape_.cols, true);
                this->dY = new cppTensor<dtype>(outputs.shape_.rows, outputs.shape_.cols, true);
            }
            else
            {
                this->loss = new cppTensor<dtype>(outputs.shape_.rows, outputs.shape_.cols);
                this->Y = new cppTensor<dtype>(outputs.shape_.rows, outputs.shape_.cols);
                this->dY = new cppTensor<dtype>(outputs.shape_.rows, outputs.shape_.cols);
            }
        }
        this->loss->zeros();
        this->Y->zeros();
        this->dY->zeros();

        if (outputs.is_cuda_ && labels.is_cuda_)
        {
            cppTensor_Functions::softmax_gpu(dY, outputs);
            cppTensor_Functions::copy_gpu(Y, *dY);

            cppTensor_Functions::log_gpu(Y);
            cppTensor_Functions::minus_gpu(Y);
            cppTensor_Functions::deriv_softmax_gpu(*dY, *loss, *Y, labels);
        }
        else
        {
            cppTensor_Functions::softmax_cpu(dY, outputs);
            cppTensor_Functions::copy_cpu(Y, *dY);
        
            cppTensor_Functions::log_cpu(Y);
            cppTensor_Functions::minus_cpu(Y);
            cppTensor_Functions::deriv_softmax_cpu(*dY, *loss, *Y, labels);
        }
        //loss->is_owner_=false;
        return *loss;
    }

    virtual void backward_impl(cppLinear<dtype>& fc, cppRnn<dtype>& rnn) override
    {
        cppTensor<dtype> O = fc.backward(*dY);
        rnn.backward(O);
    }
private:
    cppTensor<dtype>* dY{nullptr};
    cppTensor<dtype>* Y{nullptr};
    cppTensor<dtype>* loss{nullptr};
};
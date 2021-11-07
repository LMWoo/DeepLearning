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
        SAFE_DELETE(this->loss_)
        SAFE_DELETE(this->Y_)
        SAFE_DELETE(this->dY_)
    }

    cppTensor<dtype>& dY()
    {
        return *dY_;
    }

protected:
    virtual const cppTensor<dtype>& operator_impl(const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) override
    {
        if (this->loss_ == nullptr && this->Y_ == nullptr && this->dY_ == nullptr)
        {
            if (outputs.is_cuda_ && labels.is_cuda_)
            {
                this->loss_ = new cppTensor<dtype>(outputs.shape_.rows, outputs.shape_.cols, true);
                this->Y_ = new cppTensor<dtype>(outputs.shape_.rows, outputs.shape_.cols, true);
                this->dY_ = new cppTensor<dtype>(outputs.shape_.rows, outputs.shape_.cols, true);
            }
            else
            {
                this->loss_ = new cppTensor<dtype>(outputs.shape_.rows, outputs.shape_.cols);
                this->Y_ = new cppTensor<dtype>(outputs.shape_.rows, outputs.shape_.cols);
                this->dY_ = new cppTensor<dtype>(outputs.shape_.rows, outputs.shape_.cols);
            }
        }
        this->loss_->zeros();
        this->Y_->zeros();
        this->dY_->zeros();

        if (outputs.is_cuda_ && labels.is_cuda_)
        {
            cppTensor_Functions::softmax_gpu(dY_, outputs);
            cppTensor_Functions::copy_gpu(Y_, *dY_);

            cppTensor_Functions::log_gpu(Y_);
            cppTensor_Functions::minus_gpu(Y_);
            cppTensor_Functions::deriv_softmax_gpu(*dY_, *loss_, *Y_, labels);
        }
        else
        {
            cppTensor_Functions::softmax_cpu(dY_, outputs);
            cppTensor_Functions::copy_cpu(Y_, *dY_);
        
            cppTensor_Functions::log_cpu(Y_);
            cppTensor_Functions::minus_cpu(Y_);
            cppTensor_Functions::deriv_softmax_cpu(*dY_, *loss_, *Y_, labels);
        }
        //loss->is_owner_=false;
        return *loss_;
    }

    virtual void backward_impl() override
    {
        
    }

private:
    cppTensor<dtype>* dY_{nullptr};
    cppTensor<dtype>* Y_{nullptr};
    cppTensor<dtype>* loss_{nullptr};
};
#pragma once

#include "cppNN.hpp"

template<typename dtype>
class cppLinear : public cppNN<dtype>
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
    cppLinear(const cppTensorType& W, int output_size, int input_size)
    {
        this->params["W"] = new cppTensor<dtype>(output_size, input_size);
        this->params["b"] = new cppTensor<dtype>(output_size, 1);
        
        this->rParams["W"] = new cppTensor<dtype>(output_size, input_size);
        this->rParams["b"] = new cppTensor<dtype>(output_size, 1);

        this->params["dW"] = new cppTensor<dtype>(output_size, input_size);
        this->params["db"] = new cppTensor<dtype>(output_size, 1);

        this->rParams["dW"] = new cppTensor<dtype>(output_size, input_size);
        this->rParams["db"] = new cppTensor<dtype>(output_size, 1);

        this->O = new cppTensor<dtype>(input_size, 1);

        copy(this->params["W"], W);
    }

    ~cppLinear()
    {
        
    }

    virtual void cuda_impl() override
    {
        O->cuda();

        mapIntCppTensorIter mapIntIter;
        mapStrCppTensorIter mapStrIter;

        for (mapStrIter = this->params.begin(); mapStrIter != this->params.end(); ++mapStrIter)
        {
            if (mapStrIter->second)
            {
                mapStrIter->second->cuda();
            }
        }

        for (mapStrIter = this->rParams.begin(); mapStrIter != this->rParams.end(); ++mapStrIter)
        {
            if (mapStrIter->second)
            {
                mapStrIter->second->cuda();
            }
        }
    }

    virtual void cpu_impl() override
    {
        O->cpu();

        mapIntCppTensorIter mapIntIter;
        mapStrCppTensorIter mapStrIter;

        for (mapStrIter = this->params.begin(); mapStrIter != this->params.end(); ++mapStrIter)
        {
            if (mapStrIter->second)
            {
                mapStrIter->second->cpu();
            }
        }

        for (mapStrIter = this->rParams.begin(); mapStrIter != this->rParams.end(); ++mapStrIter)
        {
            if (mapStrIter->second)
            {
                mapStrIter->second->cpu();
            }
        }
    }

protected:
    virtual mapStrCppTensor parameters_impl() override
    {
        for (auto iter = this->params.begin(); iter != this->params.end(); ++iter)
        {
            this->rParams[iter->first]->shape_.rows = this->params[iter->first]->shape_.rows;
            this->rParams[iter->first]->shape_.cols = this->params[iter->first]->shape_.cols;

            if (this->is_cuda_)
            {
                this->rParams[iter->first]->dev_data_ = this->params[iter->first]->dev_data_;
            }
            else
            {
                this->rParams[iter->first]->data_ = this->params[iter->first]->data_;
            }
            
            this->rParams[iter->first]->is_owner_ = false;
        }

        return this->rParams;
    }

    virtual cppTensor<dtype> forward_impl(const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev) override
    {
        return cppTensor<dtype>();
    }

    virtual cppTensor<dtype> forward_impl(const cppTensor<dtype>& x) override
    {
        copy(this->O, x);
        return matMul(*this->params["W"], x, this->use_sharedMemory) + *this->params["b"];
    }

    virtual cppTensor<dtype> backward_impl(cppTensor<dtype>& dY) override
    {
        copy(this->params["dW"], transpose_matMul(dY, *this->O));
        copy(this->params["db"], dY);

        *this->O = matMul(transpose(*this->params["W"]), dY, this->use_sharedMemory);

        return *this->O;
    }

private:
    mapStrCppTensor params;
    mapStrCppTensor rParams;
    
    cppTensorType* O;
};
#pragma once

#include "cppOptimizer.hpp"

template<typename dtype>
class cppAdagrad : public cppOptimizer<dtype>
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
    cppAdagrad(mapStrCppTensor in_params, double lr)
    {   
        this->lr = lr;
        for (auto iter = in_params.begin(); iter != in_params.end(); ++iter)
        {
            size_t rows = in_params[iter->first]->shape_.rows;
            size_t cols = in_params[iter->first]->shape_.cols;
            if (iter->first.c_str()[0] != 'd')
            {
                std::string key = "m" + iter->first;
                if (in_params[iter->first]->is_cuda_)
                {
                    this->mem[key] = new cppTensor<dtype>(rows, cols);
                    this->mem[key]->cuda();
                }
                else
                {
                    this->mem[key] = new cppTensor<dtype>(rows, cols);
                }
            }
            this->params[iter->first] = new cppTensor<dtype>();
            this->params[iter->first]->is_owner_ = false;
            this->params[iter->first]->shape_.rows = rows;
            this->params[iter->first]->shape_.cols = cols;
            this->params[iter->first]->is_cuda_ = in_params[iter->first]->is_cuda_;

            if (params[iter->first]->is_cuda_)
            {
                this->params[iter->first]->dev_data_ = in_params[iter->first]->dev_data_;
            }
            else
            {
                this->params[iter->first]->data_ = in_params[iter->first]->data_;
            }
        }
    }

    ~cppAdagrad()
    {
        // mapStrCppTensorIter mapStrIter;
        // for (mapStrIter = this->params.begin(); mapStrIter != this->params.end(); ++mapStrIter)
        // {
        //     SAFE_DELETE(mapStrIter->second);
        // }
    }

    void test()
    {
        printf("adgrad test\n");
    }
protected:
    virtual void zero_grad_impl()
    {
        this->params["dFC_W"]->zeros();
        this->params["dfc_b"]->zeros();
        this->params["dU"]->zeros();
        this->params["dW"]->zeros();
        this->params["dV"]->zeros();
        this->params["db"]->zeros();
        this->params["dc"]->zeros();
    }

    virtual void step_impl()
    {
        clip(this->params["dU"], -5.0, 5.0);
        clip(this->params["dW"], -5.0, 5.0);
        clip(this->params["dV"], -5.0, 5.0);
        clip(this->params["db"], -5.0, 5.0);
        clip(this->params["dc"], -5.0, 5.0);
        clip(this->params["dFC_W"], -5.0, 5.0);
        clip(this->params["dfc_b"], -5.0, 5.0);
        
        optimizer(this->params["U"], this->mem["mU"], *this->params["dU"], -lr);
        optimizer(this->params["W"], this->mem["mW"], *this->params["dW"], -lr);
        optimizer(this->params["V"], this->mem["mV"], *this->params["dV"], -lr);
        optimizer(this->params["b"], this->mem["mb"], *this->params["db"], -lr);
        optimizer(this->params["c"], this->mem["mc"], *this->params["dc"], -lr);
        optimizer(this->params["FC_W"], this->mem["mFC_W"], *this->params["dFC_W"], -lr);
        optimizer(this->params["fc_b"], this->mem["mfc_b"], *this->params["dfc_b"], -lr);
    }

private:
    mapStrCppTensor params;
    mapStrCppTensor mem;

    double lr{0.0};
};
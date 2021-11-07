#pragma once

#include "cppOptimizer.hpp"

template<typename dtype>
class cppAdagrad : public cppOptimizer<dtype>
{
public:
    using mapParameters = std::unordered_map<std::string, cppTensor<dtype>*>;
    using mapModuleParameters = std::unordered_map<std::string, mapParameters>;

public:
    cppAdagrad(mapModuleParameters in_params, double lr)
    {   
        this->lr = lr;
        for (auto key = in_params.begin(); key != in_params.end(); ++key)
        {
            std::string module_name = key->first;
            for (auto iter = in_params[module_name].begin(); iter != in_params[module_name].end(); ++iter)
            {
                std::string param_name = iter->first;
                size_t rows = in_params[key->first][iter->first]->shape_.rows;
                size_t cols = in_params[key->first][iter->first]->shape_.cols;

                if (param_name.c_str()[0] == 'd')
                {
                    param_name = param_name.substr(1, param_name.length());

                    this->dParams[module_name][param_name] = new cppTensor<dtype>();
                    this->dParams[module_name][param_name]->is_owner_ = false;
                    this->dParams[module_name][param_name]->shape_.rows = rows;
                    this->dParams[module_name][param_name]->shape_.cols = cols;
                    this->dParams[module_name][param_name]->is_cuda_ = in_params[key->first][iter->first]->is_cuda_;
                    this->dParams[module_name][param_name]->dev_data_ = in_params[key->first][iter->first]->dev_data_;
                    this->dParams[module_name][param_name]->data_ = in_params[key->first][iter->first]->data_;
                }
                else
                {
                    this->mem[module_name][param_name] = new cppTensor<dtype>(rows, cols);
                    if (in_params[key->first][iter->first]->is_cuda_)
                    {
                        this->mem[module_name][param_name]->cuda();
                    }

                    this->params[module_name][param_name] = new cppTensor<dtype>();
                    this->params[module_name][param_name]->is_owner_ = false;
                    this->params[module_name][param_name]->shape_.rows = rows;
                    this->params[module_name][param_name]->shape_.cols = cols;
                    this->params[module_name][param_name]->is_cuda_ = in_params[key->first][iter->first]->is_cuda_;
                    this->params[module_name][param_name]->dev_data_ = in_params[key->first][iter->first]->dev_data_;
                    this->params[module_name][param_name]->data_ = in_params[key->first][iter->first]->data_;
                }
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
        printf("adagrad test\n");
    }
protected:
    virtual void zero_grad_impl()
    {
        for (auto key = this->dParams.begin(); key != this->dParams.end(); ++key)
        {
            std::string module_name = key->first;
            for (auto iter = this->dParams[module_name].begin(); iter != this->dParams[module_name].end(); ++iter)
            {
                std::string param_name = iter->first;
                this->dParams[module_name][param_name]->zeros();
            }
        }
    }

    virtual void step_impl()
    {   
        for (auto key = this->dParams.begin(); key != this->dParams.end(); ++key)
        {
            std::string module_name = key->first;
            for (auto iter = this->dParams[module_name].begin(); iter != this->dParams[module_name].end(); ++iter)
            {
                std::string param_name = iter->first;
                clip(this->dParams[module_name][param_name], -5.0, 5.0);
            }
        }

        for (auto key = this->dParams.begin(); key != this->dParams.end(); ++key)
        {
            std::string module_name = key->first;
            for (auto iter = this->dParams[module_name].begin(); iter != this->dParams[module_name].end(); ++iter)
            {
                std::string param_name = iter->first;
                clip(this->dParams[module_name][param_name], -5.0, 5.0);

                optimizer(this->params[module_name][param_name], this->mem[module_name][param_name], *this->dParams[module_name][param_name], -lr);
            }
        }
    }

private:
    mapModuleParameters params;
    mapModuleParameters dParams;
    mapModuleParameters mem;

    double lr{0.0};
};
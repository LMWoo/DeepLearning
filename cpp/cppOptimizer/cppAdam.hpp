#pragma once

#include "cppOptimizer.hpp"

template<typename dtype>
class cppAdam : public cppOptimizer<dtype>
{
public:
    using mapParameters = std::unordered_map<std::string, cppTensor<dtype>*>;
    using mapModuleParameters = std::unordered_map<std::string, mapParameters>;
    
public:
    cppAdam(const mapModuleParameters& parameters, double lr)
    {

    }

    ~cppAdam()
    {

    }

protected:
    virtual void zero_grad_impl()
    {

    }

    virtual void step_impl()
    {

    }
};
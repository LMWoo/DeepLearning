#pragma once

#include "cppTensor/cppTensor.hpp"
#include "cppTensor/cppTensor_Utils.hpp"
#include "cppTensor/cppTensor_Functions.hpp"

using namespace cppTensor_Functions;

template<typename dtype>
class cppOptimizer
{
public:
    cppOptimizer()
    {
        
    }

    virtual ~cppOptimizer()
    {

    }

public:
    void zero_grad()
    {
        zero_grad_impl();
    }

    void step()
    {
        step_impl();
    }

protected:
    virtual void zero_grad_impl() = 0;

    virtual void step_impl() = 0;
};
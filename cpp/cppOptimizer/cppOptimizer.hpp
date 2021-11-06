#pragma once

#include "cppTensor/cppTensor.hpp"
#include "cppTensor/cppTensor_Utils.hpp"
#include "cppTensor/cppTensor_Functions.hpp"

using namespace cppTensor_Functions;

template<typename dtype>
class cppOptimizer
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
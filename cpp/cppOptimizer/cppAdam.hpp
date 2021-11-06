#pragma once

#include "cppOptimizer.hpp"

template<typename dtype>
class cppAdam : public cppOptimizer<dtype>
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
    cppAdam(const mapStrCppTensor& parameters, double lr)
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
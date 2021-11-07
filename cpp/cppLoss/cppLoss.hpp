#pragma once

#include "cppTensor/cppTensor.hpp"
#include "cppTensor/cppTensor_Utils.hpp"
#include "cppTensor/cppTensor_Functions.hpp"
#include "cppModules/cppRnn.hpp"
#include "cppModules/cppLinear.hpp"

using namespace cppTensor_Functions;

template<typename dtype>
class cppLoss
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
    cppLoss()
    {
        
    }

    virtual ~cppLoss()
    {

    }

public:
    const cppTensor<dtype>& operator()(const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels)
    {
        return operator_impl(outputs, labels);
    }
    
    void backward()
    {
        backward_impl();
    }
    
protected:
    virtual const cppTensor<dtype>& operator_impl(const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) = 0;
    virtual void backward_impl() = 0;
};
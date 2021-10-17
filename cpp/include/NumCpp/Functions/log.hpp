#pragma once

#include <NumCpp/Core/Internal/StlAlgorithms.hpp>
#include <NumCpp/NdArray.hpp>

#include <cmath>
#include <complex>

namespace nc
{
    template<typename dtype>
    auto log(dtype inValue) noexcept 
    {
        return std::log(inValue);
    }

     template<typename dtype>
    auto log(const NdArray<dtype>& inArray) 
    {
        NdArray<decltype(log(dtype{0}))> returnArray(inArray.shape());
        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> auto
            { 
                return log(inValue);
            });
        return returnArray;
    }
}

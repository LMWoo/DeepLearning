#pragma once

#include <NumCpp/Core/Internal/StlAlgorithms.hpp>
#include <NumCpp/NdArray.hpp>

#include <cmath>
#include <complex>

namespace nc
{
    template<typename dtype>
    auto tanh(dtype inValue) noexcept
    {
        return std::tanh(inValue);
    }

    template<typename dtype>
    auto tanh(const NdArray<dtype>& inArray)
    {
        NdArray<decltype(tanh(dtype{0}))> returnArray(inArray.shape());
        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), 
            [](dtype inValue) noexcept -> auto
            {
                return tanh(inValue);
            });
        returnArray.autoMemoryOff();
        return returnArray;
    }
}
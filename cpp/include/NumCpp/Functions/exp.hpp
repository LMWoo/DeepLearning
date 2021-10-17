#pragma once

#include <NumCpp/Core/Internal/StlAlgorithms.hpp>
#include <NumCpp/NdArray.hpp>

#include <cmath>
#include <complex>

namespace nc
{
    template<typename dtype>
    auto exp(dtype inValue) noexcept
    {
        return std::exp(inValue);
    }

    template<typename dtype>
    auto exp(const NdArray<dtype>& inArray)
    {
        NdArray<decltype(exp(dtype{0}))> returnArray(inArray.shape());

        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> auto
            {
                return exp(inValue);
            });
        return returnArray;
    }
}
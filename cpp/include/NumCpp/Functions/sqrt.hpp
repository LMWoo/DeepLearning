#pragma once

#include <NumCpp/Core/Internal/StlAlgorithms.hpp>
#include <NumCpp/NdArray.hpp>

#include <cmath>
#include <complex>

namespace nc
{
    template<typename dtype>
    auto sqrt(dtype inValue) noexcept
    {
        return std::sqrt(inValue);
    }

    template<typename dtype>
    auto sqrt(const NdArray<dtype>& inArray)
    {
        NdArray<decltype(sqrt(dtype{0}))> returnArray(inArray.shape());
        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(),
            [](dtype inValue) noexcept -> auto
            {
                return sqrt(inValue);
            });

        return returnArray;
    }
}
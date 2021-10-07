#pragma once

#include "NumCpp/NdArray.hpp"

namespace nc
{
    template<typename dtype>
    NdArray<dtype> dot(const NdArray<dtype>& inArray1, const NdArray<dtype>& inArray2)
    {
        return inArray1.dot(inArray2);
    }
}
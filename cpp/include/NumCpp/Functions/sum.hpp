#pragma once

#include <NumCpp/Core/Types.hpp>
#include <NumCpp/NdArray.hpp>

namespace nc
{
    template<typename dtype>
    NdArray<dtype> sum(const NdArray<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        return inArray.sum(inAxis);
    }
}
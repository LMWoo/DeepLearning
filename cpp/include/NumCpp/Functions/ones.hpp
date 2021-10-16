#pragma once

#include <NumCpp/Core/Types.hpp>
#include <NumCpp/Functions/full.hpp>

namespace nc
{
    template<typename dtype>
    NdArray<dtype> ones(uint32 inNumRows, uint32 inNumCols)
    {
        return full(inNumRows, inNumCols, dtype{ 1 });
    }
}
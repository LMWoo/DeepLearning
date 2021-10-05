#pragma once

#include "NumMy/Core/Types.hpp"
#include "NumMy/Functions/full.hpp"

namespace nm
{
    template<typename dtype>
    NdArray<dtype> ones(uint32 inNumRows, uint32 inNumCols)
    {
        return full(inNumRows, inNumCols, dtype{ 1 });
    }
}
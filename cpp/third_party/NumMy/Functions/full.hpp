#pragma once

#include <NumMy/Core/Shape.hpp>
#include <NumMy/Core/Types.hpp>
#include <NumMy/NdArray.hpp>

namespace nm
{
    template<typename dtype>
    NdArray<dtype> full(uint32 inNumRows, uint32 inNumCols, dtype inFillValue)
    {
        NdArray<dtype> returnArray(inNumRows, inNumCols);
        
        return returnArray;
    }
}
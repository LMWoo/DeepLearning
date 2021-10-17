#pragma once

#include <NumCpp/Core/Shape.hpp>
#include <NumCpp/Core/Types.hpp>
#include <NumCpp/NdArray.hpp>

namespace nc
{
    template<typename dtype>
    NdArray<dtype> full(uint32 inNumRows, uint32 inNumCols, dtype inFillValue)
    {
        NdArray<dtype> returnArray(inNumRows, inNumCols);
        returnArray.fill(inFillValue);
        returnArray.autoMemoryOff();
        return returnArray;
    }
}
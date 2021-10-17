#pragma once

#include <NumCpp/Core/Shape.hpp>
#include <NumCpp/Core/Types.hpp>
#include <NumCpp/Functions/full.hpp>
#include <NumCpp/NdArray.hpp>

namespace nc
{
    template<typename dtype>
    NdArray<dtype> zeros(uint32 inNumRows, uint32 inNumCols)
    {
        NdArray<dtype> returnArray = full(inNumRows, inNumCols, dtype{0});
        return returnArray;
    }
}
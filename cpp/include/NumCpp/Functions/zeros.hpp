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
        PRINT_STR("Alloc Memory Call by NdArray<dtype> zeros(uint32 inNumRows, uint32 inNumCols)");
        return full(inNumRows, inNumCols, dtype{0});
    }
}
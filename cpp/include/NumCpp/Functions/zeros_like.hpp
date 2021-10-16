#pragma once

#include <NumCpp/NdArray.hpp>

namespace nc
{
    template<typename dtypeOut, typename dtype>
    NdArray<dtypeOut> zeros_like(const NdArray<dtype>& inArray)
    {
        PRINT_STR("Alloc Memory Call by NdArray<dtypeOut> zeros_like(const NdArray<dtype>& inArray)");
        NdArray<dtypeOut> returnArray(inArray.shape());
        returnArray.zeros();
        return returnArray;
    }
}
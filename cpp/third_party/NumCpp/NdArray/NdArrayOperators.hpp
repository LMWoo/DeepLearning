#pragma once

#include <NumCpp/NdArray/NdArrayCore.hpp>

namespace nc
{
    template<typename dtype>
    std::ostream& operator<<(std::ostream& inOStream, const NdArray<dtype> inArray)
    {
        inOStream << inArray.str();
        return inOStream;
    }   
}
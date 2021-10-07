#pragma once

#include <NumCpp/NdArray/NdArrayCore.hpp>

namespace nc
{
    // template<typename dtype>
    // NdArray<bool> operator!=(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    // {
    //     if (lhs.shape() != rhs.shape())
    //     {

    //     }

    //     const auto notEqualTo = [](dtype lhs, dtype rhs) noexcept -> bool
    //     {
    //         return !utils::essentiallyEqual(lhs, rhs);
    //     }

    //     NdArray<bool> returnArray(lhs.shape());

    //     stl_algorithms::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), returnArray.begin(), notEqualTo);

    //     return returnArray;
    // }
    template<typename dtype>
    std::ostream& operator<<(std::ostream& inOStream, const NdArray<dtype> inArray)
    {
        inOStream << inArray.str();
        return inOStream;
    }   
}
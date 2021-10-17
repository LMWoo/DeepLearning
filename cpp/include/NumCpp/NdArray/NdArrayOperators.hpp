#pragma once

#include <NumCpp/Core/Internal/StlAlgorithms.hpp>
#include <NumCpp/NdArray/NdArrayCore.hpp>

namespace nc
{
    template<typename dtype>
    NdArray<dtype> operator/(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        if (lhs.shape() != rhs.shape())
        {
            printf("Array dimensions do not match.\n");
        }

        NdArray<dtype> returnArray(lhs.shape());
        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), returnArray.begin(), std::divides<dtype>());

        return returnArray;
    }

    template<typename dtype>
    NdArray<dtype> operator/(const NdArray<dtype>& lhs, dtype rhs)
    {
        const auto function = [rhs](dtype value) -> dtype
        {
            return value / rhs;
        };

        NdArray<dtype> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    template<typename dtype>
    NdArray<dtype> operator/(dtype lhs, const NdArray<dtype>& rhs)
    {
        const auto function = [lhs](dtype value) -> dtype
        {
            return lhs / value;
        };

        NdArray<dtype> returnArray(rhs.shape());

        stl_algorithms::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        return returnArray;
    }

    template<typename dtype>
    NdArray<dtype> operator*(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        if (lhs.shape() != rhs.shape())
        {
            printf("Array dimensions do not match.\n");
        }

        NdArray<dtype> returnArray(lhs.shape());
        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), returnArray.begin(), std::multiplies<dtype>());
        
        returnArray.autoMemoryOff();
        return returnArray;
    }

    template<typename dtype>
    NdArray<dtype> operator*(const NdArray<dtype>& lhs, dtype rhs)
    {
        const auto function = [rhs](dtype value) -> dtype
        {
            return value * rhs;
        };

        NdArray<dtype> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        returnArray.autoMemoryOff();
        return returnArray;
    }

    template<typename dtype>
    NdArray<dtype> operator*(dtype lhs, const NdArray<dtype>& rhs)
    {
        NdArray<dtype> returnArray = rhs * lhs;
        returnArray.autoMemoryOff();
        return returnArray;
    }

    template<typename dtype>
    NdArray<dtype> operator-(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        if (lhs.shape() != rhs.shape())
        {
            printf("Array dimensions do not match.\n");
        }

        NdArray<dtype> returnArray(lhs.shape());
        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), returnArray.begin(), std::minus<dtype>());
        returnArray.autoMemoryOff();
        return returnArray;
    }

    template<typename dtype>
    NdArray<dtype> operator-(const NdArray<dtype>& lhs, dtype rhs)
    {
        const auto function = [rhs](dtype value) -> dtype
        {
            return value - rhs;
        };

        NdArray<dtype> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        returnArray.autoMemoryOff();
        return returnArray;
    }

    template<typename dtype>
    NdArray<dtype> operator-(dtype lhs, const NdArray<dtype>& rhs)
    {
        const auto function = [lhs](dtype value) -> dtype
        {
            return lhs - value;
        };

        NdArray<dtype> returnArray(rhs.shape());

        stl_algorithms::transform(rhs.cbegin(), rhs.cend(), returnArray.begin(), function);

        returnArray.autoMemoryOff();
        return returnArray;
    }

    template<typename dtype>
    NdArray<dtype> operator-(const NdArray<dtype>& inArray)
    {
        const auto function = [](dtype value) -> dtype
        {
            return -value;
        };

        auto returnArray = NdArray<dtype>(inArray.shape());
        stl_algorithms::transform(inArray.cbegin(), inArray.cend(), returnArray.begin(), function);

        returnArray.autoMemoryOff();
        return returnArray;
    }

    template<typename dtype>
    NdArray<dtype> operator+(const NdArray<dtype>& lhs, const NdArray<dtype>& rhs)
    {
        if (lhs.shape() != rhs.shape())
        {
            printf("Array dimensions do not match.\n");
        }

        NdArray<dtype> returnArray(lhs.shape());
        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), returnArray.begin(), std::plus<dtype>());
        returnArray.autoMemoryOff();

        return returnArray;
    }

    template<typename dtype>
    NdArray<dtype> operator+(const NdArray<dtype>& lhs, dtype rhs)
    {
        const auto function = [rhs](dtype value) -> dtype
        {
            return value + rhs;
        };

        NdArray<dtype> returnArray(lhs.shape());

        stl_algorithms::transform(lhs.cbegin(), lhs.cend(), returnArray.begin(), function);

        returnArray.autoMemoryOff();
        return returnArray;
    }

    template<typename dtype>
    NdArray<dtype> operator+(dtype lhs, const NdArray<dtype>& rhs)
    {
        NdArray<dtype> returnArray = rhs + lhs;

        returnArray.autoMemoryOff();
        return returnArray;
    }

    template<typename dtype>
    std::ostream& operator<<(std::ostream& inOStream, const NdArray<dtype> inArray)
    {
        inOStream << inArray.str();
        return inOStream;
    }   
}
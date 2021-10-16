#pragma once

#include <NumCpp/Core/Shape.hpp>
#include <NumCpp/NdArray.hpp>
#include <NumCpp/Random/generator.hpp>

#include <algorithm>
#include <random>

namespace nc
{
    namespace random
    {
        template<typename dtype>
        NdArray<dtype> rand(dtype inLow, dtype inHigh, const Shape& inShape)
        {
            PRINT_STR("Alloc Memory Call by NdArray<dtype> rand(dtype inLow, dtype inHigh, const Shape& inShape)");
            NdArray<dtype> returnArray(inShape);

            std::uniform_real_distribution<dtype> dist(inLow, inHigh - DtypeInfo<dtype>::epsilon());

            std::for_each(returnArray.begin(), returnArray.end(), [&dist](dtype& value) -> void
            {
                value = dist(generator_);
            });

            return returnArray;
        }
    }
}
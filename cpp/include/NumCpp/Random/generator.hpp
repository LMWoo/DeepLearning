#pragma once

#include <NumCpp/Core/Types.hpp>

#include <random>

namespace nc
{
    namespace random
    {
        static std::mt19937_64 generator_;

        inline void seed(uint32 inSeed)
        {
            generator_.seed(inSeed);
        }
    }
}
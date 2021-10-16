#pragma once

#include <complex>
#include <limits>

namespace nc
{
    template<typename dtype>
    class DtypeInfo
    {
    public:
        static constexpr dtype epsilon() noexcept
        {
            return std::numeric_limits<dtype>::epsilon();
        }
    };

    template<typename dtype>
    class DtypeInfo<std::complex<dtype>>
    {
    public:
        static constexpr std::complex<dtype> epsilon() noexcept
        {
            return { DtypeInfo<dtype>::epsilon(), DtypeInfo<dtype>::epsilon() };
        }
    };
}
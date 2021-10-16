#pragma once

#include <complex>
#include <type_traits>

namespace nc
{
    template<class A, class B>
    constexpr bool is_same_v = std::is_same<A, B>::value;
}
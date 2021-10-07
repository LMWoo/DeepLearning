#pragma once

#include <algorithm>

namespace nc
{
    namespace stl_algorithms
    {
        template<class InputIt, class OutputIt>
        OutputIt copy(InputIt first, InputIt last, OutputIt destination) noexcept
        {
            return std::copy(first, last, destination);
        }

        template<class ForwardIt, class T>
        void fill(ForwardIt first, ForwardIt last, const T& value) noexcept
        {
            return std::fill(first, last, value);
        }
        
    }
}
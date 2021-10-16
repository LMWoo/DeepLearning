#pragma once

#include <algorithm>

namespace nc
{
    namespace stl_algorithms
    {
        template<class InputIt, class OutputIt, class UnaryOperation>
        OutputIt transform(InputIt first, InputIt last, OutputIt destination, UnaryOperation unaryFunction)
        {
            return std::transform(first, last, destination, unaryFunction);
        }

        template<class InputIt1, class InputIt2, class OutputIt, class BinaryOperation>
        OutputIt transform(InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt destination, BinaryOperation unaryFunction)
        {
            return std::transform(first1, last1, first2, destination, unaryFunction);
        }

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
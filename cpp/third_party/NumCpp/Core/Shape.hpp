#pragma once

#include <NumCpp/Core/Types.hpp>

#include <iostream>
#include <string>

namespace nc
{
    class Shape
    {
    public:
        uint32 rows{ 0 };
        uint32 cols{ 0 };

        constexpr Shape() = default;

        constexpr explicit Shape(uint32 inSquareSize) noexcept :
            rows(inSquareSize),
            cols(inSquareSize)
        {}

        constexpr Shape(uint32 inRows, uint32 inCols) noexcept :
            rows(inRows),
            cols(inCols)
        {}       

        uint32 size() const noexcept
        {
            return rows * cols;
        }
    };
}
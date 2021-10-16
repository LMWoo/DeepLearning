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

        bool operator==(const Shape& inOtherShape) const noexcept
        {
            return rows == inOtherShape.rows && cols == inOtherShape.cols;
        }

        bool operator!=(const Shape& inOtherShape) const noexcept
        {
            return !(*this == inOtherShape);
        }

        std::string str() const
        {
            std::string out;
            out += "[ shape : rows ";
            out += std::to_string(rows);
            out += " cols ";
            out += std::to_string(cols);
            out += " ]\n";

            return out;
        }

        void print() const
        {
            std::cout << str();
        }

        uint32 size() const noexcept
        {
            return rows * cols;
        }
    };
}
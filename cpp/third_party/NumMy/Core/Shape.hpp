#pragma once

#include <NumMy/Core/Types.hpp>

#include <iostream>
#include <string>

namespace nm
{
    class Shape
    {
    public:
        uint32 rows{ 0 };
        uint32 cols{ 0 };

        constexpr Shape(uint32 inRows, uint32 inCols) noexcept :
            rows(inRows),
            cols(inCols)
        {}       

    };
}
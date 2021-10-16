#pragma once

#include <complex>
#include <sstream>
#include <string>

namespace nc
{
    namespace utils
    {
        template<typename dtype>
        std::string value2str(dtype inValue)
        {
            std::stringstream ss;
            ss << inValue;
            return ss.str();
        }
    }
}
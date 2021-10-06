#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace nc
{
    namespace pybindInterface
    {
        template<typename dtype>
        using pbArray = pybind11::array_t<dtype, pybind11::array::c_style>;
        using pbArrayGeneric = pybind11::array;

        template<typename dtype>
        pbArrayGeneric nc2pybind(const NdArray<dtype>& inArray)
        {
            const Shape inShape = inArray.shape();
            const std::vector<pybind11::ssize_t> shape{ static_cast<pybind11::ssize_t>(inShape.rows),
                static_cast<pybind11::ssize_t>(inShape.cols)};
            const std::vector<pybind11::ssize_t> strides{ static_cast<pybind11::ssize_t>(inShape.cols * sizeof(dtype)),
                static_cast<pybind11::ssize_t>(sizeof(dtype)) };
            return pbArrayGeneric(shape, strides, inArray.data());
        }
    }
}
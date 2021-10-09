#pragma once

#include <NumCpp/Core/Shape.hpp>
#include <NumCpp/NdArray.hpp>
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
        NdArray<dtype> pybind2nc(const pbArray<dtype>& numpyArray)
        {
            const auto dataPtr = numpyArray.data();
            switch(numpyArray.ndim())
            {
                case 0:
                {
                    return NdArray<dtype>(dataPtr, 0, 0);
                }
                case 1:
                {
                    const uint32 size = static_cast<uint32>(numpyArray.size());
                    return NdArray<dtype>(dataPtr, 1, size);
                }
                case 2:
                {
                    const uint32 numRows = static_cast<uint32>(numpyArray.shape(0));
                    const uint32 numCols = static_cast<uint32>(numpyArray.shape(1));
                    return NdArray<dtype>(dataPtr, numRows, numCols);
                }
                default:
                {
                    return NdArray<dtype>(0, 0);
                }
            }
        }

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
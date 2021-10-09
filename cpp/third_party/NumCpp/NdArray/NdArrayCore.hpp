#pragma once
#include <NumCpp/Core/Internal/StlAlgorithms.hpp>
#include <NumCpp/Core/Internal/TypeTraits.hpp>
#include <NumCpp/Core/Shape.hpp>
#include <NumCpp/Core/Types.hpp>
#include <NumCpp/NdArray/NdArrayIterator.hpp>
#include <NumCpp/Utils/value2str.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <memory>

namespace nc
{
    template<typename dtype>
    class NdArray
    {
    private:
        using pbArray = pybind11::array_t<dtype, pybind11::array::c_style>;
        using pbArrayGeneric = pybind11::array;
    
    public:
        using value_type = dtype;
        using pointer = dtype*;
        using const_pointer = const dtype*;
        using reference = dtype&;
        using const_reference = const dtype&;
        using size_type = uint32;
        
        NdArray()
        {
            printf("NdArray()\n");
        }

        NdArray(size_type inNumRows, size_type inNumCols) :
            shape_(inNumRows, inNumCols),
            size_(inNumRows * inNumCols)
        {
            newArray();
        }

        NdArray(const_pointer inPtr, uint32 numRows, uint32 numCols) :
            shape_(numRows, numCols),
            size_(numRows * numCols)
        {
            newArray();
            std::copy(inPtr, inPtr + size_, array_);
        }

        explicit NdArray(const Shape& inShape) :
            shape_(inShape),
            size_(shape_.size())
        {
            newArray();
        }

        explicit NdArray(pbArray& numpyArray)
        {
            // *this = pybind2nc(numpyArray);
            pybind2nc(numpyArray);
        }

        void memoryFree()
        {
            if (ownsPtr_ && array_ != nullptr)
            {
                free(array_);
                array_=nullptr;
            }
        }
        ~NdArray()
        {
            // shape_.rows = shape_.cols = 0;
            // size_ = 0;
            // ownsPtr_ = false;

            // deleteArray();
            // if (ownsPtr_)
            // {
            //     ownsPtr_=false;
            //     deleteArray();
            // }
        }

        // NdArray<dtype> pybind2nc(const pbArray& numpyArray)
        void pybind2nc(const pbArray& numpyArray)
        {
            const auto dataPtr = numpyArray.data();

            switch(numpyArray.ndim())
            {
                // case 0:
                // {
                //     return NdArray<dtype>(dataPtr, 0, 0);
                // }
                // case 1:
                // {
                //     const uint32 size = static_cast<uint32>(numpyArray.size());
                //     return NdArray<dtype>(dataPtr, 1, size);
                // }
                case 2:
                {
                    const uint32 numRows = static_cast<uint32>(numpyArray.shape(0));
                    const uint32 numCols = static_cast<uint32>(numpyArray.shape(1));
                    shape_ = Shape(numRows, numCols);
                    size_ = shape_.size();
                    newArray();
                    std::copy(dataPtr, dataPtr + size_, array_);
                }
                default:
                {
                    break;
                }
                // default:
                // {
                //     return NdArray<dtype>(0, 0);
                // }
            }
        }

        Shape shape() const noexcept
        {
            return shape_;
        }

        pointer data() noexcept
        {
            return array_;
        }

        const_pointer data() const noexcept
        {
            return array_;
        }

        NdArray<dtype>& ones() noexcept
        {
            fill(dtype{ 1 });
            return *this;
        }

        NdArray<dtype>& zeros() noexcept
        {
            fill(dtype{ 0 });
            return *this;
        }

        pointer begin() noexcept
        {
            return array_;
        }

        pointer end() noexcept
        {
            return begin() + size_;
        }

        pointer cbegin() const noexcept
        {
            return array_;
        }

        pointer cbegin(size_type inRow) const
        {
            return cbegin() + (inRow * shape_.cols);
        }

        pointer cend() const noexcept
        {
            return cbegin() + size_;
        }

        pointer cend(size_type inRow) const
        {
            return cbegin(inRow) + shape_.cols;
        }

        NdArray<dtype> dot(const NdArray<dtype>& inOtherArray) const
        {
            if (shape_.cols == inOtherArray.shape_.rows)
            {
                NdArray<dtype> returnArray(shape_.rows, inOtherArray.shape_.cols);
                auto otherArrayT = inOtherArray.transpose();
                for (uint32 i = 0; i < shape_.rows; ++i)
                {
                    for (uint32 j = 0; j < otherArrayT.shape_.rows; ++j)
                    {
                        returnArray(i, j) = std::inner_product(otherArrayT.cbegin(j), otherArrayT.cend(j), cbegin(i), dtype{0});
                    }
                }
                return returnArray;
            }

            return NdArray<dtype>(0, 0);
        }

        NdArray<dtype> dot_debug(const NdArray<dtype>& inOtherArray) const
        {
            if (shape_.cols == inOtherArray.shape_.rows)
            {
                NdArray<dtype> returnArray(shape_.rows, inOtherArray.shape_.cols);
                auto otherArrayT = inOtherArray.transpose();
                for (uint32 i = 0; i < shape_.rows; ++i)
                {
                    for (uint32 j = 0; j < otherArrayT.shape_.rows; ++j)
                    {
                        returnArray(i, j) = std::inner_product(otherArrayT.cbegin(j), otherArrayT.cend(j), cbegin(i), dtype{0});
                    }
                }
                return returnArray;
            }

            return NdArray<dtype>(0, 0);
        }

        NdArray<dtype>& fill(value_type inFillValue) noexcept
        {
            std::fill(begin(), end(), inFillValue);
            return *this;
        }

        NdArray<dtype> transpose() const
        {
            NdArray<dtype> transArray(shape_.cols, shape_.rows);
            for (uint32 row = 0; row < shape_.rows; ++row)
            {
                for (uint32 col = 0; col < shape_.cols; ++col)
                {
                    transArray(col, row) = operator()(row, col);
                }
            }

            return transArray;
        }

        NdArray<dtype>& operator=(const NdArray<dtype>& rhs) noexcept
        {
            if (rhs.size_ > 0)
            {
                newArray(rhs.shape_);
                std::copy(rhs.cbegin(), rhs.cend(), begin());
            }
            return *this;
        }

        NdArray<dtype>& operator=(value_type inValue) noexcept
        {
            if (array_ != nullptr)
            {
                std::fill(begin(), end(), inValue);
            }
            return *this;
        }

        reference operator()(int32 inRowIndex, int32 inColIndex) noexcept
        {
            if (inRowIndex < 0)
            {
                inRowIndex += shape_.rows;
            }
            if (inColIndex < 0)
            {
                inColIndex += shape_.cols;
            }
            return array_[inRowIndex * shape_.cols + inColIndex];
        }

        const_reference operator()(int32 inRowIndex, int32 inColIndex) const noexcept
        {
            if (inRowIndex < 0)
            {
                inRowIndex += shape_.rows;
            }
            if (inColIndex < 0)
            {
                inColIndex += shape_.cols;
            }
            return array_[inRowIndex * shape_.cols + inColIndex];
        }
        
        std::string str() const
        {
            std::string out;
            out += "[";
            for (uint32 row = 0; row < shape_.rows; ++row)
            {
                out += "[";
                for (uint32 col = 0; col < shape_.cols; ++col)
                {
                    out += utils::value2str(operator()(row, col)) + ", ";
                }
                if (row == shape_.rows - 1)
                {
                    out += "]";
                }
                else
                {
                    out += "]\n";
                }
            }
            out += "]\n";
            return out;
        }

        void print() const
        {
            std::cout << *this;
        }
    
    private:
        Shape shape_{0, 0};
        size_type size_{0};
        pointer array_{nullptr};
        bool ownsPtr_{false};
        void deleteArray() noexcept
        {
            // if (ownsPtr_ && array_ != nullptr)
            // {
            //     printf("free\n");
            //     free(array_);
            //     array_=nullptr;
            // }

            // shape_.rows = shape_.cols = 0;
            // size_ = 0;
            // ownsPtr_ = false;
        }

        void newArray()
        {
            if (size_ > 0)
            {
                array_ = (pointer)malloc(size_ * sizeof(dtype));
                ownsPtr_ = true;
            }
        }

        void newArray(const Shape& inShape)
        {
            deleteArray();

            shape_ = inShape;
            size_ = inShape.size();
            newArray();
        }
    };
}
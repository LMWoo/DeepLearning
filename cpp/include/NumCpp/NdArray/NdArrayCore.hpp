#pragma once

#include <NumCpp/Core/DtypeInfo.hpp>
#include <NumCpp/Core/Internal/StlAlgorithms.hpp>
#include <NumCpp/Core/Internal/TypeTraits.hpp>
#include <NumCpp/Core/Shape.hpp>
#include <NumCpp/Core/Types.hpp>
#include <NumCpp/Utils/value2str.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "test_gpu.h"
#include <vector>
#include <memory>

#define NDARRAY_DEBUG

#if defined(NDARRAY_DEBUG)
#define PRINT_STR(x) do { \
    printf("%s\n", (x)); \
} while(0)
#define PRINT_INT(x) do { \
    printf("%d\n", (x)); \
} while(0)
#define PRINT_PTR(x) do { \
    printf("%p\n", (x)); \
} while(0)
#else
#define PRINT_STR(x)
#define PRINT_INT(x)
#define PRINT_PTR(x)
#endif

namespace nc
{
    bool rnn_debug_start=false;
    namespace memory
    {
        std::vector<double*> memoryDouble;
        std::vector<int*> memoryInt;
        
        void push(double* array)
        {
            memoryDouble.push_back(array);
        }

        void push(int* array)
        {
            memoryInt.push_back(array);
        }

        void memoryClean()
        {
            for (size_t i = 0; i < memoryDouble.size(); ++i)
            {
                if (memoryDouble[i] != nullptr)
                {
                    free(memoryDouble[i]);
                    memoryDouble[i] = nullptr;
                }
            }

            memoryDouble.clear();

            for (size_t i = 0; i < memoryInt.size(); ++i)
            {
                if (memoryInt[i] != nullptr)
                {
                    free(memoryInt[i]);
                    memoryInt[i] = nullptr;
                }
            }

            memoryInt.clear();
        }
    }
    
        
    template<typename dtype>
    class NdArray
    {
    public:
        using value_type = dtype;
        using pointer = dtype*;
        using const_pointer = const dtype*;
        using reference = dtype&;
        using const_reference = const dtype&;
        using size_type = uint32;
        
        using pbArray = pybind11::array_t<dtype, pybind11::array::c_style>;
        using pbArrayGeneric = pybind11::array;


        NdArray() = default;

        NdArray(size_type inNumRows, size_type inNumCols) :
            shape_(inNumRows, inNumCols),
            size_(inNumRows * inNumCols)
        {
            autoDelete = true;
            newArray();
        }

        NdArray(const std::initializer_list<dtype>& inList) :
            shape_(1, static_cast<uint32>(inList.size())),
            size_(shape_.size())
        {
            autoDelete = true;
            newArray();
            if (size_ > 0)
            {
                stl_algorithms::copy(inList.begin(), inList.end(), begin());
            }
        }

        NdArray(const_pointer inPtr, uint32 numRows, uint32 numCols) :
            shape_(numRows, numCols),
            size_(numRows * numCols)
        {
            autoDelete = true;
            newArray();
            std::copy(inPtr, inPtr + size_, array_);
        }

        explicit NdArray(const Shape& inShape) :
            shape_(inShape),
            size_(shape_.size())
        {
            autoDelete = true;
            newArray();
        }

        explicit NdArray(pbArray& numpyArray)
        {
            autoDelete = true;
            pybind2nc(numpyArray);
        }

        void autoMemoryOff()
        {
            autoDelete = false;
            memory::push(this->array_);
        }

        ~NdArray()
        {
            if (autoDelete && array_ != nullptr)
            {
                if (rnn_debug_start)
                {
                    PRINT_STR("~NdArray()");
                    PRINT_PTR(array_);
                }
                deleteArray();
            }
        }

        void pybind2nc(const pbArray& numpyArray)
        {
            const auto dataPtr = numpyArray.data();

            switch(numpyArray.ndim())
            {
                case 0:
                {
                    break;
                }
                case 1:
                {
                    const uint32 size = static_cast<uint32>(numpyArray.size());
                    shape_ = Shape(1, size);
                    size_ = shape_.size();
                    newArray();
                    std::copy(dataPtr, dataPtr + size_, array_);
                    break;
                }
                case 2:
                {
                    const uint32 numRows = static_cast<uint32>(numpyArray.shape(0));
                    const uint32 numCols = static_cast<uint32>(numpyArray.shape(1));
                    shape_ = Shape(numRows, numCols);
                    size_ = shape_.size();
                    newArray();
                    std::copy(dataPtr, dataPtr + size_, array_);
                    break;
                }
                default:
                {
                    break;
                }
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

        NdArray<dtype> sum(Axis inAxis = Axis::NONE) const
        {
            switch(inAxis)
            {
                case Axis::NONE:
                {
                    NdArray<dtype> returnArray = {std::accumulate(cbegin(), cend(), dtype{0})};
                    returnArray.autoMemoryOff();
                    return returnArray;
                }
                default:
                {
                    return {};
                }
            }
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
                returnArray.autoMemoryOff();
                return returnArray;
            }

            NdArray<dtype> returnArray(1, 1);
            returnArray.autoMemoryOff();
            return returnArray;
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
            transArray.autoMemoryOff();
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
                    if (col > 0 && col % 4 == 0)
                    {
                        out += '\n';
                    }
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
            std::cout << str();
        }

        void initialize()
        {
            newArray();
        }
    private:
        Shape shape_{0, 0};
        size_type size_{0};
        pointer array_{nullptr};
        bool autoDelete{true};

        void deleteArray() noexcept
        {
            if (array_)
            {
                free(array_);
                array_=nullptr;
            }
            
            shape_.rows = shape_.cols = 0;
            size_ = 0;
        }

        void newArray()
        {
            if (size_ > 0)
            {
                array_ = (pointer)malloc(size_ * sizeof(dtype));
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
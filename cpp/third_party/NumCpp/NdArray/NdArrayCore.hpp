#pragma once

#include <NumCpp/Core/Internal/StlAlgorithms.hpp>
#include <NumCpp/Core/Internal/TypeTraits.hpp>
#include <NumCpp/Core/Shape.hpp>
#include <NumCpp/Core/Types.hpp>
#include <NumCpp/NdArray/NdArrayIterator.hpp>
// #include <NumCpp/PythonInterface/PybindInterface.hpp>
#include <NumCpp/Utils/value2str.hpp>

#include <memory>

namespace nc
{
    template<typename dtype, class Allocator = std::allocator<dtype>>
    class NdArray
    {
    private:

        static_assert(is_same_v<dtype, typename Allocator::value_type>, "value_type and Allocator::value_type must match");

        using AllocType = typename std::allocator_traits<Allocator>::template rebind_alloc<dtype>;
        using AllocTraits = std::allocator_traits<AllocType>;

    public:
        using value_type = dtype;
        using allocator_type = Allocator;
        using pointer = typename AllocTraits::pointer;
        using const_pointer = typename AllocTraits::const_pointer;
        using reference = dtype&;
        using const_reference = const dtype&;
        using size_type = uint32;
        using difference_type = typename AllocTraits::difference_type;

        using iterator = NdArrayIterator<dtype, pointer, difference_type>;
        using const_iterator = NdArrayConstIterator<dtype, const_pointer, difference_type>;

        NdArray(size_type inNumRows, size_type inNumCols) :
            shape_(inNumRows, inNumCols),
            size_(inNumRows * inNumCols)
            {
                newArray();
            }

        template<typename Bool, std::enable_if_t<is_same_v<Bool, bool>, int> = 0>
        NdArray(pointer inPtr, uint32 numRows, uint32 numCols, Bool takeOwnership) noexcept:
            shape_(numRows, numCols),
            size_(numRows * numCols),
            array_(inPtr),
            ownsPtr_(takeOwnership)
        {}
        
        explicit NdArray(const Shape& inShape) :
            shape_(inShape),
            size_(shape_.size())
            {
                newArray();
            }
        
        // explicit NdArray(pybindInterface::pbArray<dtype>& numpyArray)
        // {
        //     *this = pybind2nc(numpyArray);
        // }

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
            return cbegin() += size_;
        }

        pointer cend(size_type inRow) const
        {
            return cbegin(inRow) + shape_.cols;
        }
        // const_iterator cbegin() const noexcept
        // {
        //     return const_iterator(array_);
        // }

        // const_iterator cbegin(size_type inRow) const
        // {
        //     if (inRow >= shape_.rows)
        //     {

        //     }

        //     return cbegin() += (inRow * shape_.cols);
        // }

        // const_iterator cend() const noexcept
        // {
        //     return cbegin() += size_;
        // }

        // const_iterator cend(size_type inRow) const
        // {
        //     if (inRow >= shape_.rows)
        //     {

        //     }

        //     return cbegin(inRow) += shape_.cols;
        // }

        NdArray<dtype> dot(const NdArray<dtype>& inOtherArray) const
        {
            // if (shape_ == inOtherArray.shape_ && (shape_.rows == 1 || shape_.cols == 1))
            // {
            //     dtype dotProduct = std::inner_product(cbegin(), cend(), inOtherArray.cbegin(), dtype{0});
            //     NdArray<dtype> returnArray = {dotProduct};
            //     return returnArray;
            // }
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
            std::fill(array_, array_ + size_, inFillValue);
            // std::fill(begin(), end(), inFillValue);
            // stl_algorithms::fill(begin(), end(), inFillValue);
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

        Shape shape() const noexcept
        {
            return shape_;
        }

        pointer data() noexcept
        {
            return array_;
        }

        pointer begin() noexcept
        {
            return array_;
        }

        pointer end() noexcept
        {
            return begin() + size_;
        }
        // iterator begin() noexcept
        // {
        //     return iterator(array_);
        // }

        // iterator end() noexcept
        // {
        //     return begin() += size_;
        // }


        NdArray<dtype>& operator=(const NdArray<dtype>& rhs) noexcept
        {
            // if (&rhs != this)
            {
                if (rhs.size_ > 0)
                {
                    newArray(rhs.shape_);
                    // endianess_ = rhs.endianess_;

                    stl_algorithms::copy(rhs.cbegin(), rhs.cend(), begin());
                }
            }

            return *this;
        }

        NdArray<dtype>& operator=(value_type inValue) noexcept
        {
            if (array_ != nullptr)
            {
                std::fill(array_, array_ + size_, inValue);
            }

            return *this;
        }

        // NdArray<dtype>& operator=(NdArray<dtype>&& rhs) noexcept
        // {
        //     if (&rhs != this)
        //     {
        //         deleteArray();
        //         shape_ = rhs.shape_;
        //         size_ = rhs.size_;
        //         array_ = rhs.array_;
        //         ownsPtr_ = rhs.ownsPtr_;

        //         rhs.shape_.rows = rhs.shape_.cols = rhs.size_ = 0;
        //         rhs.array_ = nullptr;
        //         rhs.ownsPtr_ = false;
        //     }

        //     return *this;
        // }

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
    private:
        allocator_type allocator_{};
        Shape shape_{0, 0};
        size_type size_{ 0 };
        // Endian endianess_{Endian::NATIVE};
        pointer array_{ nullptr };
        bool ownsPtr_{ false };

        void deleteArray() noexcept
        {
            if (ownsPtr_ && array_ != nullptr)
            {
                allocator_.deallocate(array_, size_);
            }

            array_ = nullptr;
            shape_.rows = shape_.cols = 0;
            size_ = 0;
            ownsPtr_ = false;
            // endianess_ = Endian::NATIVE;
        }

        void newArray()
        {
            if (size_ > 0)
            {
                array_ = allocator_.allocate(size_);
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
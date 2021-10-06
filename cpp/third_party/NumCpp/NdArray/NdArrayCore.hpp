#pragma once

#include <NumCpp/Core/Internal/StlAlgorithms.hpp>
#include <NumCpp/Core/Shape.hpp>
#include <NumCpp/Core/Types.hpp>
#include <NumCpp/NdArray/NdArrayIterator.hpp>
#include <NumCpp/Utils/value2str.hpp>

#include <memory>

namespace nc
{
    template<typename dtype, class Allocator = std::allocator<dtype>>
    class NdArray
    {
    private:

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

        NdArray(size_type inNumRows, size_type inNumCols) :
            shape_(inNumRows, inNumCols),
            size_(inNumRows * inNumCols)
            {
                newArray();
            }
        
        explicit NdArray(const Shape& inShape) :
            shape_(inShape),
            size_(shape_.size())
            {
                newArray();
            }

        void newArray()
        {
            if (size_ > 0)
            {
                array_ = allocator_.allocate(size_);
                ownsPtr_ = true;
            }
        }

        NdArray<dtype>& fill(value_type inFillValue) noexcept
        {
            std::fill(array_, array_ + size_, inFillValue);
            // std::fill(begin(), end(), inFillValue);
            // stl_algorithms::fill(begin(), end(), inFillValue);
            return *this;
        }

        NdArray<dtype>& ones() noexcept
        {
            fill(dtype{ 1 });
            return *this;
        }

        Shape shape() const noexcept
        {
            return shape_;
        }

        pointer data() noexcept
        {
            return array_;
        }

        iterator begin() noexcept
        {
            return iterator(array_);
        }

        iterator end() noexcept
        {
            return begin() += size_;
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
    private:
        Shape shape_{0, 0};
        size_type size_{ 0 };
        allocator_type allocator_{};
        pointer array_{ nullptr };
        bool ownsPtr_{ false };
    };
}
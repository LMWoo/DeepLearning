#pragma once

#include <NumCpp/Core/Types.hpp>

#include <iterator>

namespace nc
{
    template<typename dtype, typename PointerType, typename DifferenceType>
    class NdArrayConstIterator
    {
    private:
        using self_type = NdArrayConstIterator<dtype, PointerType, DifferenceType>;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = dtype;
        using pointer = PointerType;
        using reference = const value_type&;
        using difference_type = DifferenceType;

        NdArrayConstIterator() = default;

        explicit NdArrayConstIterator(pointer ptr) :
            ptr_(ptr)
            {
                if (ptr == nullptr)
                {

                }
            }

        self_type& operator+=(const difference_type offset) noexcept
        {
            ptr_ += offset;
            return *this;
        }
    
    private:
        pointer ptr_{nullptr};
    };

    template<typename dtype, typename PointerType, typename DifferenceType>
    class NdArrayIterator : public NdArrayConstIterator<dtype, PointerType, DifferenceType>
    {
    private:
        using MyBase = NdArrayConstIterator<dtype, PointerType, DifferenceType>;
        using self_type = NdArrayIterator<dtype, PointerType, DifferenceType>;
    
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = dtype;
        using pointer = PointerType;
        using reference = value_type&;
        using difference_type = DifferenceType;

        using MyBase::MyBase;

        self_type& operator+=(const difference_type offset) noexcept
        {
            MyBase::operator+=(offset);
            return *this;
        }
    };
}
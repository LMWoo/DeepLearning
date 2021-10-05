#pragma once

#include <NumMy/Core/Shape.hpp>
#include <NumMy/Core/Types.hpp>

#include <memory>

namespace nm
{
    template<typename dtype, class Allocator = std::allocator<dtype>>
    class NdArray
    {
    private:

        using AllocType = typename std::allocator_traits<Allocator>::template rebind_alloc<dtype>;
        using AllocTraits = std::allocator_traits<AllocType>;

    public:
        using allocator_type = Allocator;
        using pointer = typename AllocTraits::pointer;
        using size_type = uint32;


        NdArray(size_type inNumRows, size_type inNumCols) :
            shape_(inNumRows, inNumCols),
            size_(inNumRows * inNumCols)
            {
                newArray();
            }
        
        void newArray()
        {
            if (size_ > 0)
            {
                array_ = allocator_.allocate(size_);
                ownsPtr_ = true;
                printf("newArray\n");
            }
        }

    private:
        Shape shape_{0, 0};
        size_type size_{ 0 };
        allocator_type allocator_{};
        pointer array_{ nullptr };
        bool ownsPtr_{ false };
    };
}
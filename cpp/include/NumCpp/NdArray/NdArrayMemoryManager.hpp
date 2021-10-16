#pragma once

#include <NumCpp/NdArray/NdArrayCore.hpp>
#include <vector>

namespace nc
{
    namespace memory
    {
        // template<typename dtype> class NdArray;
        // using NdArrayDoubleP = NdArray<double>*;
        // using NdArrayIntP = NdArray<int>*;

        // // template<typename dtype>
        // // void NdArray<dtype>::memoryFree();

        // template<typename arrayType>
        // class MemoryManagerBase
        // {
        // private:
        //     std::vector<arrayType> vec;

        // public:
        //     void push(arrayType array)
        //     {
        //         vec.push_back(array);
        //     }

        //     void memoryFree()
        //     {
        //         for (int i = 0; i < vec.size(); ++i)
        //         {
        //             vec[i]->memoryFree();
        //         }
        //     }
        // };

        // static MemoryManagerBase<NdArrayDoubleP> MMDouble;
        // static MemoryManagerBase<NdArrayIntP> MMInt;

        // static void push(NdArrayDoubleP arrayP)
        // {
        //     MMDouble.push(arrayP);
        // }

        // static void push(NdArrayIntP arrayP)
        // {
        //     MMInt.push(arrayP);
        // }

        // static void memoryFree()
        // {
        //     MMDouble.memoryFree();
        //     MMInt.memoryFree();
        // }        
    }
}
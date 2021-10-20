#pragma once

#include "NumTest.hpp"

template<typename dtype>
class cppRnn
{
public:
    using numpyArray = pybind11::array_t<dtype, pybind11::array::c_style>;
    using numpyArrayGeneric = pybind11::array;
    using numTestType = numTest<dtype>;
public:
    cppRnn(double lr, const numTestType& U, const numTestType& W, const numTestType& V, const numTestType& FC_W,
        size_t seq_length, size_t input_size, size_t hidden_size, size_t num_classes)
    {
        this->lr = lr;
        this->seq_length = seq_length;
        this->hidden_size = hidden_size;

        this->U = U;
        this->W = W;
        this->V = V;

        this->FC_W = FC_W;
    }

    ~cppRnn()
    {
        PRINT_DEBUG("call by ~cppRnn() this %p data %p\n", &this->U, this->U.data_);
        PRINT_DEBUG("call by ~cppRnn() this %p data %p\n", &this->W, this->W.data_);
        PRINT_DEBUG("call by ~cppRnn() this %p data %p\n", &this->V, this->V.data_);
        PRINT_DEBUG("call by ~cppRnn() this %p data %p\n", &this->FC_W, this->FC_W.data_);
        this->U.data_ = nullptr;
        this->W.data_ = nullptr;
        this->V.data_ = nullptr;
        this->FC_W.data_ = nullptr;
    }

    void test()
    {
        printf("cppRnn test\n");
    }
private:
    double lr{0.0};
    size_t seq_length{0};
    size_t hidden_size{0};

    numTestType U;
    numTestType W;
    numTestType V;

    numTestType FC_W;

};
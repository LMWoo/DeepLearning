#pragma once

#include "NumTest.hpp"
#include <unordered_map>
#include <tuple>

template<typename dtype>
class cppRnn
{
public:
    using numpyArray = pybind11::array_t<dtype, pybind11::array::c_style>;
    using numpyArrayGeneric = pybind11::array;
    using numTestType = numTest<dtype>;
    using numTestTypeMap = std::unordered_map<int, numTest<dtype>>;
    using numTestTypeMapDoubleIter = std::unordered_map<int, numTest<double>>::iterator;

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
        // PRINT_DEBUG("call by ~cppRnn() this %p data %p\n", &this->U, this->U.data_);
        // PRINT_DEBUG("call by ~cppRnn() this %p data %p\n", &this->W, this->W.data_);
        // PRINT_DEBUG("call by ~cppRnn() this %p data %p\n", &this->V, this->V.data_);
        // PRINT_DEBUG("call by ~cppRnn() this %p data %p\n", &this->FC_W, this->FC_W.data_);
        this->U.data_ = nullptr;
        this->U.dev_data_=nullptr;
        this->W.data_ = nullptr;
        this->W.dev_data_=nullptr;
        this->V.data_ = nullptr;
        this->V.dev_data_=nullptr;
        this->FC_W.data_ = nullptr;
        this->FC_W.dev_data_=nullptr;

        numTestTypeMapDoubleIter mapIter;
        for (mapIter = this->S.begin(); mapIter != this->S.end(); ++mapIter)
        {
            mapIter->second.data_=nullptr;
            mapIter->second.dev_data_=nullptr;
        }
    }

    void cuda()
    {
        is_cuda_=true;
    }

    void cpu()
    {
        is_cuda_=false;
    }

    void test()
    {
        PRINT_DEBUG("cppRnn test\n");
    }

public:
    void forward(numTest<dtype>& result,  std::vector<numTest<dtype>>& x,  numTest<dtype>& hprev)
    {
#ifdef NUMTEST_DEBUG
        // printf("cpp forward x print start\n");
        // for (int i = 0; i < x.size(); ++i)
        // {
        //     x[i].cpu();
        //     x[i].print();
        //     x[i].cuda();

        //     x_is_cuda=x[i].is_cuda_;
        // }
        // printf("cpp forward x print end\n");

        // printf("cpp forward hprev print start\n");
        // hprev.cpu();
        // hprev.print();
        // hprev.cuda();
        // printf("cpp forward hprev print end\n");

        // printf("cpp forward result print start\n");
        // result.cpu();
        // result.print();
        // result.cuda();
        // printf("cpp forward result print end\n");
#endif
        bool x_is_cuda = false;
        
        for (int i = 0; i < x.size(); ++i)
        {
            x_is_cuda=x[i].is_cuda_;
        }

        if ((this->is_cuda_ && x_is_cuda) && hprev.is_cuda_)
        {
            forward_gpu(result, x, hprev);
        }
        else
        {
            forward_cpu(result, x, hprev);
        }

        // result.data_=nullptr;
        // hprev.data_=nullptr;
        
        for (int i = 0; i < x.size(); ++i)
        {
            // x[i].print();
            x[i].data_=nullptr;
            x[i].dev_data_=nullptr;
        }
    }

private:
    void forward_gpu(numTest<dtype>& result, const std::vector<numTest<dtype>>& x, const numTest<dtype>& hprev)
    {
        PRINT_DEBUG("call by forward_gpu() start\n");
        S[-1] = hprev;
        PRINT_DEBUG("call by forward_gpu() end\n");
    }

    void forward_cpu(numTest<dtype>& result, const std::vector<numTest<dtype>>& x, const numTest<dtype>& hprev)
    {
        PRINT_DEBUG("call by forward_cpu() start\n");
        S[-1] = hprev;
        PRINT_DEBUG("call by forward_cpu() end\n");
    }


private:
    double lr{0.0};
    bool is_cuda_{false};

    size_t seq_length{0};
    size_t hidden_size{0};

    numTestType U;
    numTestType W;
    numTestType V;

    numTestType FC_W;

    numTestTypeMap S;
};
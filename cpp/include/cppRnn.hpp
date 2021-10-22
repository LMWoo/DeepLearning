#pragma once

#include "NumTest.hpp"
#include "NumTest_Functions.hpp"
#include <unordered_map>
#include <tuple>

template<typename dtype>
class cppRnn
{
public:
    using numpyArray = pybind11::array_t<dtype, pybind11::array::c_style>;
    using numpyArrayGeneric = pybind11::array;
    using numTestType = numTest<dtype>;
    using numTestTypeMap = std::unordered_map<int, numTest<dtype>*>;
    using numTestTypeMapDoubleIter = std::unordered_map<int, numTest<double>*>::iterator;

public:
    cppRnn(double lr, const numTestType& U, const numTestType& W, const numTestType& V, const numTestType& FC_W,
        int seq_length, int input_size, int hidden_size, int num_classes)
    {
        this->lr = lr;
        this->seq_length = seq_length;
        this->hidden_size = hidden_size;
        
        this->U = new numTest<dtype>(hidden_size, input_size);
        this->W = new numTest<dtype>(hidden_size, hidden_size);
        this->V = new numTest<dtype>(hidden_size, hidden_size);
        this->b = new numTest<dtype>(hidden_size, 1);
        this->c = new numTest<dtype>(hidden_size, 1);

        numTest_Functions::copy_cpu(this->U, U);
        numTest_Functions::copy_cpu(this->W, W);
        numTest_Functions::copy_cpu(this->V, V);

        this->FC_W = new numTest<dtype>(num_classes, hidden_size);
        this->fc_b = new numTest<dtype>(num_classes, 1);
        numTest_Functions::copy_cpu(this->FC_W, FC_W);
        
        for (int i = -1; i < seq_length; ++i)
        {
            this->S[i] = new numTest<dtype>(hidden_size, 1);
        }

        for (int i = 0; i < seq_length; ++i)
        {
            this->X[i] = new numTest<dtype>(input_size, 1);
        }

        for (int i = 0; i < seq_length + 2; ++i)
        {
            this->A[i] = new numTest<dtype>(hidden_size, 1);
        }

        for (int i = 0; i < seq_length + 1; ++i)
        {
            this->O[i] = new numTest<dtype>(hidden_size, 1);
        }
    }

    ~cppRnn()
    {
        if (U)
        {
            delete U;
            U = nullptr;
        }
        if (W)
        {
            delete W;
            W = nullptr;
        }
        if (V)
        {
            delete V;
            V = nullptr;
        }
        if (b)
        {
            delete b;
            b = nullptr;
        }
        if (c)
        {
            delete c;
            c = nullptr;
        }

        if (FC_W)
        {
            delete FC_W;
            FC_W = nullptr;
        }
        if (fc_b)
        {
            delete fc_b;
            fc_b = nullptr;
        }

        numTestTypeMapDoubleIter mapIter;

        for (mapIter = this->S.begin(); mapIter != this->S.end(); ++mapIter)
        {
            if (mapIter->second)
            {
                delete mapIter->second;
                mapIter->second=nullptr;
            }
        }

        for (mapIter = this->X.begin(); mapIter != this->X.end(); ++mapIter)
        {
            if (mapIter->second)
            {
                delete mapIter->second;
                mapIter->second=nullptr;
            }
        }

        for (mapIter = this->A.begin(); mapIter != this->A.end(); ++mapIter)
        {
            if (mapIter->second)
            {
                delete mapIter->second;
                mapIter->second=nullptr;
            }
        }

        for (mapIter = this->O.begin(); mapIter != this->O.end(); ++mapIter)
        {
            if (mapIter->second)
            {
                delete mapIter->second;
                mapIter->second=nullptr;
            }
        }
    }

    void cuda()
    {
        is_cuda_=true;

        U->cuda();
        W->cuda();
        V->cuda();
        b->cuda();
        c->cuda();
        FC_W->cuda();
        fc_b->cuda();

        numTestTypeMapDoubleIter mapIter;
        for (mapIter = this->S.begin(); mapIter != this->S.end(); ++mapIter)
        {
            if (mapIter->second)
            {
                mapIter->second->cuda();
            }
        }

        for (mapIter = this->X.begin(); mapIter != this->X.end(); ++mapIter)
        {
            if (mapIter->second)
            {
                mapIter->second->cuda();
            }
        }

        for (mapIter = this->A.begin(); mapIter != this->A.end(); ++mapIter)
        {
            if (mapIter->second)
            {
                mapIter->second->cuda();
            }
        }

        for (mapIter = this->O.begin(); mapIter != this->O.end(); ++mapIter)
        {
            if (mapIter->second)
            {
                mapIter->second->cuda();
            }
        }
    }

    void cpu()
    {
        is_cuda_=false;

        U->cpu();
        W->cpu();
        V->cpu();
        b->cpu();
        c->cpu();
        FC_W->cpu();
        fc_b->cpu();

        numTestTypeMapDoubleIter mapIter;
        for (mapIter = this->S.begin(); mapIter != this->S.end(); ++mapIter)
        {
            if (mapIter->second)
            {
                mapIter->second->cpu();
            }
        }

        for (mapIter = this->X.begin(); mapIter != this->X.end(); ++mapIter)
        {
            if (mapIter->second)
            {
                mapIter->second->cpu();
            }
        }

        for (mapIter = this->A.begin(); mapIter != this->A.end(); ++mapIter)
        {
            if (mapIter->second)
            {
                mapIter->second->cpu();
            }
        }

        for (mapIter = this->O.begin(); mapIter != this->O.end(); ++mapIter)
        {
            if (mapIter->second)
            {
                mapIter->second->cpu();
            }
        }
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
            NumTest_Utils::time_start();
            forward_gpu(result, x, hprev);
            NumTest_Utils::time_end();
        }
        else
        {
            NumTest_Utils::time_start();
            forward_cpu(result, x, hprev);
            NumTest_Utils::time_end();
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
        numTest_Functions::copy_gpu(S[-1], hprev);

        for (int t = 0; t < seq_length; ++t)
        {
            numTest_Functions::transpose_gpu(X[t], x[t]);
            numTest_Functions::dot_gpu(A[seq_length], *U, *X[t]);
            numTest_Functions::dot_gpu(A[seq_length + 1], *W, *S[t - 1]);
            numTest_Functions::add_gpu(A[t], *A[seq_length], *A[seq_length + 1]);
            numTest_Functions::add_gpu(A[t], *A[t], *b);
            numTest_Functions::tanh_gpu(S[t], *A[t]);
            numTest_Functions::dot_gpu(O[seq_length], *V, *S[t]);
            numTest_Functions::add_gpu(O[t], *O[seq_length], *c);
        }

        numTest_Functions::dot_gpu(&result, *FC_W, *O[seq_length-1]);
        numTest_Functions::add_gpu(&result, result, *fc_b);
        
        PRINT_DEBUG("call by forward_gpu() end\n");
    }

    void forward_cpu(numTest<dtype>& result, const std::vector<numTest<dtype>>& x, const numTest<dtype>& hprev)
    {
        PRINT_DEBUG("call by forward_cpu() start\n");
        numTest_Functions::copy_cpu(S[-1], hprev);
        
        for (int t = 0; t < seq_length; ++t)
        {
            numTest_Functions::transpose_cpu(X[t], x[t]);
            numTest_Functions::dot_cpu(A[seq_length], *U, *X[t]);
            numTest_Functions::dot_cpu(A[seq_length + 1], *W, *S[t - 1]);
            numTest_Functions::add_cpu(A[t], *A[seq_length], *A[seq_length + 1]);
            numTest_Functions::add_cpu(A[t], *A[t], *b);
            numTest_Functions::tanh_cpu(S[t], *A[t]);
            numTest_Functions::dot_cpu(O[seq_length], *V, *S[t]);
            numTest_Functions::add_cpu(O[t], *O[seq_length], *c);
        }

        numTest_Functions::dot_cpu(&result, *FC_W, *O[seq_length-1]);
        numTest_Functions::add_cpu(&result, result, *fc_b);

        PRINT_DEBUG("call by forward_cpu() end\n");
    }


private:
    double lr{0.0};
    bool is_cuda_{false};

    size_t seq_length{0};
    size_t hidden_size{0};

    numTestType* U;
    numTestType* W;
    numTestType* V;
    numTestType* b;
    numTestType* c;

    numTestType* FC_W;
    numTestType* fc_b;

    numTestTypeMap X;
    numTestTypeMap A;
    numTestTypeMap S;
    numTestTypeMap O;
};
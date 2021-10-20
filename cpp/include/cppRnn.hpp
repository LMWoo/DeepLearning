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
        numTest_Functions::copy_cpu(this->U, U);
        numTest_Functions::copy_cpu(this->W, W);
        numTest_Functions::copy_cpu(this->V, V);

        this->FC_W = new numTest<dtype>(num_classes, hidden_size);
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
        if (FC_W)
        {
            delete FC_W;
            FC_W = nullptr;
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
    }

    void cuda()
    {
        is_cuda_=true;

        U->cuda();
        W->cuda();
        V->cuda();
        
        FC_W->cuda();

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

    }

    void cpu()
    {
        is_cuda_=false;

        U->cpu();
        W->cpu();
        V->cpu();
        
        FC_W->cpu();

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
        numTest_Functions::copy_gpu(S[-1], hprev);
        
        // S[-1]->cpu();
        // S[-1]->print();
        // S[-1]->cuda();

        for (int t = 0; t < seq_length; ++t)
        {
            numTest_Functions::transpose_gpu(X[t], x[t]);
            numTest_Functions::dot_gpu(A[t], *U, *X[t]);


#ifdef NUMTEST_DEBUG
            // A[t]->cpu();
            
            // printf("gpu start\n");
            // A[t]->print();
            // printf("gpu end\n");
            
            // U->cpu();
            // X[t]->cpu();

            // numTest_Functions::dot_cpu(A[t], *U, *X[t]);

            // printf("cpu start\n");
            // A[t]->print();
            // printf("cpu end\n");
            // A[t]->cuda();
            // U->cuda();
            // X[t]->cuda();
#endif
        }
        
        for (int t = 0; t < seq_length; ++t)
        {
            printf("prev transpose\n");
            // x[t].cpu<dtype>();
            // x[t].print<dtype>();
            // x[t].cuda<dtype>();

            printf("after transpose\n");
            // X[t]->cpu();
            // X[t]->print();
            // X[t]->cuda();
        }


        PRINT_DEBUG("call by forward_gpu() end\n");
    }

    void forward_cpu(numTest<dtype>& result, const std::vector<numTest<dtype>>& x, const numTest<dtype>& hprev)
    {
        PRINT_DEBUG("call by forward_cpu() start\n");
        numTest_Functions::copy_cpu(S[-1], hprev);
        
        for (int t = 0; t < seq_length; ++t)
        {
            numTest_Functions::transpose_cpu(X[t], x[t]);
            numTest_Functions::dot_cpu(A[t], *U, *X[t]);
            // A[seq_length]->print();
        }

        for (int t = 0; t < seq_length; ++t)
        {
            printf("U start\n");
            U->print();
            printf("U end\n");

            printf("X[t] start\n");
            X[t]->print();
            printf("X[t] end\n");

            printf("A[t] start\n");
            A[t]->print();
            printf("A[t] end\n");
        }

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

    numTestType* FC_W;

    numTestTypeMap X;
    numTestTypeMap A;
    numTestTypeMap S;
};
#pragma once

#include "cppTensor.hpp"
#include "cppTensor_Functions.hpp"
#include <unordered_map>
#include <tuple>

template<typename dtype>
class cppRnn
{
public:
    using numpyArray = pybind11::array_t<dtype, pybind11::array::c_style>;
    using numpyArrayGeneric = pybind11::array;
    using cppTensorType = cppTensor<dtype>;
    using cppTensorTypeMap = std::unordered_map<int, cppTensor<dtype>*>;
    using cppTensorTypeMapDoubleIter = std::unordered_map<int, cppTensor<double>*>::iterator;

public:
    cppRnn(double lr, const cppTensorType& U, const cppTensorType& W, const cppTensorType& V, const cppTensorType& FC_W,
        int seq_length, int input_size, int hidden_size, int num_classes)
    {
        this->lr = lr;
        this->seq_length = seq_length;
        this->hidden_size = hidden_size;
        
        this->U = new cppTensor<dtype>(hidden_size, input_size);
        this->W = new cppTensor<dtype>(hidden_size, hidden_size);
        this->V = new cppTensor<dtype>(hidden_size, hidden_size);
        this->b = new cppTensor<dtype>(hidden_size, 1);
        this->c = new cppTensor<dtype>(hidden_size, 1);

        cppTensor_Functions::copy_cpu(this->U, U);
        cppTensor_Functions::copy_cpu(this->W, W);
        cppTensor_Functions::copy_cpu(this->V, V);

        this->FC_W = new cppTensor<dtype>(num_classes, hidden_size);
        this->fc_b = new cppTensor<dtype>(num_classes, 1);
        cppTensor_Functions::copy_cpu(this->FC_W, FC_W);
        
        for (int i = -1; i < seq_length; ++i)
        {
            this->S[i] = new cppTensor<dtype>(hidden_size, 1);
        }

        for (int i = 0; i < seq_length; ++i)
        {
            this->X[i] = new cppTensor<dtype>(input_size, 1);
        }

        for (int i = 0; i < seq_length + 2; ++i)
        {
            this->A[i] = new cppTensor<dtype>(hidden_size, 1);
        }

        for (int i = 0; i < seq_length + 1; ++i)
        {
            this->O[i] = new cppTensor<dtype>(hidden_size, 1);
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

        cppTensorTypeMapDoubleIter mapIter;

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

        cppTensorTypeMapDoubleIter mapIter;
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

        cppTensorTypeMapDoubleIter mapIter;
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
    void forward(cppTensor<dtype>& outputs,  std::vector<cppTensor<dtype>>& x,  cppTensor<dtype>& hprev)
    {
        bool x_is_cuda = false;
        
        for (int i = 0; i < x.size(); ++i)
        {
            x_is_cuda=x[i].is_cuda_;
        }

        if ((this->is_cuda_ && x_is_cuda) && hprev.is_cuda_)
        {
            cppTensor_Utils::time_start();
            forward_gpu(outputs, x, hprev);
            cppTensor_Utils::time_end();
        }
        else
        {
            cppTensor_Utils::time_start();
            forward_cpu(outputs, x, hprev);
            cppTensor_Utils::time_end();
        }

        // outputs.data_=nullptr;
        // hprev.data_=nullptr;
        
        for (int i = 0; i < x.size(); ++i)
        {
            // x[i].print();
            x[i].data_=nullptr;
            x[i].dev_data_=nullptr;
        }
    }


    void cross_entropy_loss(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels)
    {
        if ((Y.is_cuda_ && outputs.is_cuda_) && (loss.is_cuda_ && labels.is_cuda_))
        {
            cross_entropy_loss_gpu(dY, Y, loss, outputs, labels);
        }
        else
        {
            cross_entropy_loss_cpu(dY, Y, loss, outputs, labels);
        }
    }
    
private:

    void cross_entropy_loss_gpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels)
    {
        cppTensor_Functions::softmax_gpu(&dY, outputs);
        cppTensor_Functions::copy_gpu(&Y, dY);

        cppTensor_Functions::log_gpu(&Y);
        cppTensor_Functions::minus_gpu(&Y);
        cppTensor_Functions::deriv_softmax_gpu(dY, loss, Y, labels);

        // const_cast<cppTensor<dtype>&>(labels).print();
        // Y.print();
        // loss.print();
    }

    void cross_entropy_loss_cpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels)
    {
        cppTensor_Functions::softmax_cpu(&dY, outputs);
        cppTensor_Functions::copy_cpu(&Y, dY);
        
        cppTensor_Functions::log_cpu(&Y);
        cppTensor_Functions::minus_cpu(&Y);
        cppTensor_Functions::deriv_softmax_cpu(dY, loss, Y, labels);

        // const_cast<cppTensor<dtype>&>(labels).print();
        // Y.print();
        // loss.print();
    }

    void forward_gpu(cppTensor<dtype>& outputs, const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev)
    {
        PRINT_DEBUG("call by forward_gpu() start\n");
        cppTensor_Functions::copy_gpu(S[-1], hprev);

        for (int t = 0; t < seq_length; ++t)
        {
            cppTensor_Functions::transpose_gpu(X[t], x[t]);
            cppTensor_Functions::dot_gpu(A[seq_length], *U, *X[t]);
            cppTensor_Functions::dot_gpu(A[seq_length + 1], *W, *S[t - 1]);
            cppTensor_Functions::add_gpu(A[t], *A[seq_length], *A[seq_length + 1]);
            cppTensor_Functions::add_gpu(A[t], *A[t], *b);
            cppTensor_Functions::tanh_gpu(S[t], *A[t]);
            cppTensor_Functions::dot_gpu(O[seq_length], *V, *S[t]);
            cppTensor_Functions::add_gpu(O[t], *O[seq_length], *c);
        }

        cppTensor_Functions::dot_gpu(&outputs, *FC_W, *O[seq_length-1]);
        cppTensor_Functions::add_gpu(&outputs, outputs, *fc_b);
        
        PRINT_DEBUG("call by forward_gpu() end\n");
    }

    void forward_cpu(cppTensor<dtype>& outputs, const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev)
    {
        PRINT_DEBUG("call by forward_cpu() start\n");
        cppTensor_Functions::copy_cpu(S[-1], hprev);
        
        for (int t = 0; t < seq_length; ++t)
        {
            cppTensor_Functions::transpose_cpu(X[t], x[t]);
            cppTensor_Functions::dot_cpu(A[seq_length], *U, *X[t]);
            cppTensor_Functions::dot_cpu(A[seq_length + 1], *W, *S[t - 1]);
            cppTensor_Functions::add_cpu(A[t], *A[seq_length], *A[seq_length + 1]);
            cppTensor_Functions::add_cpu(A[t], *A[t], *b);
            cppTensor_Functions::tanh_cpu(S[t], *A[t]);
            cppTensor_Functions::dot_cpu(O[seq_length], *V, *S[t]);
            cppTensor_Functions::add_cpu(O[t], *O[seq_length], *c);
        }

        cppTensor_Functions::dot_cpu(&outputs, *FC_W, *O[seq_length-1]);
        cppTensor_Functions::add_cpu(&outputs, outputs, *fc_b);

        PRINT_DEBUG("call by forward_cpu() end\n");
    }

private:
    double lr{0.0};
    bool is_cuda_{false};

    size_t seq_length{0};
    size_t hidden_size{0};

    cppTensorType* U;
    cppTensorType* W;
    cppTensorType* V;
    cppTensorType* b;
    cppTensorType* c;

    cppTensorType* FC_W;
    cppTensorType* fc_b;

    cppTensorTypeMap X;
    cppTensorTypeMap A;
    cppTensorTypeMap S;
    cppTensorTypeMap O;
};
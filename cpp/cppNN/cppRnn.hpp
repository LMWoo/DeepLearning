#pragma once

#include "cppNN.hpp"

template<typename dtype>
class cppRnn : public cppNN<dtype>
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

        this->FC_W = new cppTensor<dtype>(num_classes, hidden_size);
        this->fc_b = new cppTensor<dtype>(num_classes, 1);

        // this->params.push_back(this->U);
        // this->params.push_back(this->W);
        // this->params.push_back(this->V);
        // this->params.push_back(this->b);
        // this->params.push_back(this->c);
        // this->params.push_back(this->FC_W);
        // this->params.push_back(this->fc_b);

        cppTensor_Functions::copy_cpu(this->U, U);
        cppTensor_Functions::copy_cpu(this->W, W);
        cppTensor_Functions::copy_cpu(this->V, V);

        cppTensor_Functions::copy_cpu(this->FC_W, FC_W);

        this->dU = new cppTensor<dtype>(hidden_size, input_size);
        this->dW = new cppTensor<dtype>(hidden_size, hidden_size);
        this->dV = new cppTensor<dtype>(hidden_size, hidden_size);
        this->db = new cppTensor<dtype>(hidden_size, 1);
        this->dc = new cppTensor<dtype>(hidden_size, 1);

        this->dFC_W = new cppTensor<dtype>(num_classes, hidden_size);
        this->dfc_b = new cppTensor<dtype>(num_classes, 1);

        // this->dparams.push_back(this->dU);
        // this->dparams.push_back(this->dW);
        // this->dparams.push_back(this->dV);
        // this->dparams.push_back(this->db);
        // this->dparams.push_back(this->dc);
        // this->dparams.push_back(this->dFC_W);
        // this->dparams.push_back(this->dfc_b);

        this->mU = new cppTensor<dtype>(hidden_size, input_size);
        this->mW = new cppTensor<dtype>(hidden_size, hidden_size);
        this->mV = new cppTensor<dtype>(hidden_size, hidden_size);
        this->mb = new cppTensor<dtype>(hidden_size, 1);
        this->mc = new cppTensor<dtype>(hidden_size, 1);

        this->mFC_W = new cppTensor<dtype>(num_classes, hidden_size);
        this->mfc_b = new cppTensor<dtype>(num_classes, 1);

        // this->mem.push_back(this->mU);
        // this->mem.push_back(this->mW);
        // this->mem.push_back(this->mV);
        // this->mem.push_back(this->mb);
        // this->mem.push_back(this->mc);
        // this->mem.push_back(this->mFC_W);
        // this->mem.push_back(this->mfc_b);
        
        this->dO = new cppTensor<dtype>(hidden_size, 1);
        this->dA = new cppTensor<dtype>(hidden_size, 1);
        this->dS = new cppTensor<dtype>(hidden_size, 1);
        this->dS_next = new cppTensor<dtype>(hidden_size, 1);

        for (int i = -1; i < seq_length; ++i)
        {
            this->S[i] = new cppTensor<dtype>(hidden_size, 1);
        }

        for (int i = 0; i < seq_length; ++i)
        {
            this->X[i] = new cppTensor<dtype>(input_size, 1);
        }

        for (int i = 0; i < seq_length; ++i)
        {
            this->A[i] = new cppTensor<dtype>(hidden_size, 1);
        }

        for (int i = 0; i < seq_length; ++i)
        {
            this->O[i] = new cppTensor<dtype>(hidden_size, 1);
        }
    }

    ~cppRnn()
    {
        SAFE_DELETE(U)
        SAFE_DELETE(W)
        SAFE_DELETE(V)
        SAFE_DELETE(b)
        SAFE_DELETE(c)
        SAFE_DELETE(FC_W)
        SAFE_DELETE(fc_b)
        
        SAFE_DELETE(dU)
        SAFE_DELETE(dW)
        SAFE_DELETE(dV)
        SAFE_DELETE(db)
        SAFE_DELETE(dc)
        SAFE_DELETE(dFC_W)
        SAFE_DELETE(dfc_b)

        SAFE_DELETE(dO)
        SAFE_DELETE(dA)
        SAFE_DELETE(dS)
        SAFE_DELETE(dS_next)
        
        SAFE_DELETE(mU)
        SAFE_DELETE(mW)
        SAFE_DELETE(mV)
        SAFE_DELETE(mb)
        SAFE_DELETE(mc)
        SAFE_DELETE(mFC_W)
        SAFE_DELETE(mfc_b)

        // params.clear();
        // dparams.clear();
        // mem.clear();

        cppTensorTypeMapDoubleIter mapIter;

        for (mapIter = this->S.begin(); mapIter != this->S.end(); ++mapIter)
        {
            SAFE_DELETE(mapIter->second)
        }

        for (mapIter = this->X.begin(); mapIter != this->X.end(); ++mapIter)
        {
            SAFE_DELETE(mapIter->second)
        }

        for (mapIter = this->A.begin(); mapIter != this->A.end(); ++mapIter)
        {
            SAFE_DELETE(mapIter->second)
        }

        for (mapIter = this->O.begin(); mapIter != this->O.end(); ++mapIter)
        {
            SAFE_DELETE(mapIter->second)
        }
    }

    virtual void cuda_impl() override
    {
        // for (int i = 0; i < params.size(); ++i)
        // {
        //     params[i]->cuda();
        // }

        // for (int i = 0; i < dparams.size(); ++i)
        // {
        //     dparams[i]->cuda();
        // }

        // for (int i = 0; i < dparams.size(); ++i)
        // {
        //     mem[i]->cuda();
        // }

        U->cuda();
        W->cuda();
        V->cuda();
        b->cuda();
        c->cuda();
        FC_W->cuda();
        fc_b->cuda();

        dU->cuda();
        dW->cuda();
        dV->cuda();
        db->cuda();
        dc->cuda();
        dFC_W->cuda();
        dfc_b->cuda();

        mU->cuda();
        mW->cuda();
        mV->cuda();
        mb->cuda();
        mc->cuda();
        mFC_W->cuda();
        mfc_b->cuda();

        dO->cuda();
        dA->cuda();
        dS->cuda();
        dS_next->cuda();

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

    virtual void cpu_impl() override
    {
        // for (int i = 0; i < params.size(); ++i)
        // {
        //     params[i]->cpu();
        // }

        // for (int i = 0; i < dparams.size(); ++i)
        // {
        //     dparams[i]->cpu();
        // }

        // for (int i = 0; i < dparams.size(); ++i)
        // {
        //     mem[i]->cpu();
        // }

        U->cpu();
        W->cpu();
        V->cpu();
        b->cpu();
        c->cpu();
        FC_W->cpu();
        fc_b->cpu();

        dU->cpu();
        dW->cpu();
        dV->cpu();
        db->cpu();
        dc->cpu();
        dFC_W->cpu();
        dfc_b->cpu();

        mU->cpu();
        mW->cpu();
        mV->cpu();
        mb->cpu();
        mc->cpu();
        mFC_W->cpu();
        mfc_b->cpu();

        dO->cpu();
        dA->cpu();
        dS->cpu();
        dS_next->cpu();

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

protected:

    virtual void optimizer_impl() override
    {
        if (this->is_cuda_)
        {
            cppTensor_Functions::clip_gpu(dU, -5.0, 5.0);
            cppTensor_Functions::clip_gpu(dW, -5.0, 5.0);
            cppTensor_Functions::clip_gpu(dV, -5.0, 5.0);
            cppTensor_Functions::clip_gpu(db, -5.0, 5.0);
            cppTensor_Functions::clip_gpu(dc, -5.0, 5.0);
            cppTensor_Functions::clip_gpu(dFC_W, -5.0, 5.0);
            cppTensor_Functions::clip_gpu(dfc_b, -5.0, 5.0);
        
            cppTensor_Functions::optimizer_gpu(U, mU, *dU, -lr);
            cppTensor_Functions::optimizer_gpu(W, mW, *dW, -lr);
            cppTensor_Functions::optimizer_gpu(V, mV, *dV, -lr);
            cppTensor_Functions::optimizer_gpu(b, mb, *db, -lr);
            cppTensor_Functions::optimizer_gpu(c, mc, *dc, -lr);
            cppTensor_Functions::optimizer_gpu(FC_W, mFC_W, *dFC_W, -lr);
            cppTensor_Functions::optimizer_gpu(fc_b, mfc_b, *dfc_b, -lr);
        }
        else
        {
            cppTensor_Functions::clip_cpu(dU, -5.0, 5.0);
            cppTensor_Functions::clip_cpu(dW, -5.0, 5.0);
            cppTensor_Functions::clip_cpu(dV, -5.0, 5.0);
            cppTensor_Functions::clip_cpu(db, -5.0, 5.0);
            cppTensor_Functions::clip_cpu(dc, -5.0, 5.0);
            cppTensor_Functions::clip_cpu(dFC_W, -5.0, 5.0);
            cppTensor_Functions::clip_cpu(dfc_b, -5.0, 5.0);

            cppTensor_Functions::optimizer_cpu(U, mU, *dU, -lr);
            cppTensor_Functions::optimizer_cpu(W, mW, *dW, -lr);
            cppTensor_Functions::optimizer_cpu(V, mV, *dV, -lr);
            cppTensor_Functions::optimizer_cpu(b, mb, *db, -lr);
            cppTensor_Functions::optimizer_cpu(c, mc, *dc, -lr);
            cppTensor_Functions::optimizer_cpu(FC_W, mFC_W, *dFC_W, -lr);
            cppTensor_Functions::optimizer_cpu(fc_b, mfc_b, *dfc_b, -lr);
        }
    }

    virtual void backward_impl(const cppTensor<dtype>& dY) override
    {
        dFC_W->zeros();
        dfc_b->zeros();
        dU->zeros();
        dW->zeros();
        dV->zeros();
        db->zeros();
        dc->zeros();
        dS_next->zeros();
        
        *dFC_W = matMul(dY, transpose(*O[seq_length - 1]), this->use_sharedMemory);
        copy(dfc_b, dY);

        *dO = matMul(transpose(*FC_W), dY, this->use_sharedMemory);

        *dV = matMul(*dO, transpose(*S[seq_length - 1]), this->use_sharedMemory);
        copy(dc, *dO);

        for (int t = seq_length - 1; t >= 0; --t)
        {
            *dS = matMul(transpose(*V), *dO, this->use_sharedMemory) + *dS_next;
            *dA = deriv_tanh(*S[t]) * (*dS);
            
            *dU = *dU + matMul(*dA, transpose(*X[t]), this->use_sharedMemory);
            *dW = *dW + matMul(*dA, transpose(*S[t - 1]), this->use_sharedMemory);
            *db = *db + *dA;
            *dS_next = matMul(transpose(*W), *dA, this->use_sharedMemory);
        }
    }

    virtual void cross_entropy_loss_impl(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) override
    {
        if (this->is_cuda_)
        {
            cppTensor_Functions::softmax_gpu(&dY, outputs);
            cppTensor_Functions::copy_gpu(&Y, dY);

            cppTensor_Functions::log_gpu(&Y);
            cppTensor_Functions::minus_gpu(&Y);
            cppTensor_Functions::deriv_softmax_gpu(dY, loss, Y, labels);
        }
        else
        {
            cppTensor_Functions::softmax_cpu(&dY, outputs);
            cppTensor_Functions::copy_cpu(&Y, dY);
        
            cppTensor_Functions::log_cpu(&Y);
            cppTensor_Functions::minus_cpu(&Y);
            cppTensor_Functions::deriv_softmax_cpu(dY, loss, Y, labels);
        }
    }

    virtual cppTensor<dtype> forward_impl(const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev) override
    {
        copy(S[-1], hprev);
        
        for (int t = 0; t < seq_length; ++t)
        {
            *X[t] = transpose(x[t]);
            *A[t] = matMul(*U, *X[t], this->use_sharedMemory) + matMul(*W, *S[t - 1], this->use_sharedMemory) + *b;
            *S[t] = tanh(*A[t]);
            *O[t] = matMul(*V, *S[t], this->use_sharedMemory) + *c;
        }

        return matMul(*FC_W, *O[seq_length-1], this->use_sharedMemory) + *fc_b;
    }

private:
    double lr{0.0};
    size_t seq_length{0};
    size_t hidden_size{0};

    cppTensorType* U;
    cppTensorType* W;
    cppTensorType* V;
    cppTensorType* b;
    cppTensorType* c;

    cppTensorType* FC_W;
    cppTensorType* fc_b;

    // std::vector<cppTensorType*> params;

    cppTensorType* dU;
    cppTensorType* dW;
    cppTensorType* dV;
    cppTensorType* db;
    cppTensorType* dc;
    
    cppTensorType* dFC_W;
    cppTensorType* dfc_b;

    // std::vector<cppTensorType*> dparams;

    cppTensorType* mU;
    cppTensorType* mW;
    cppTensorType* mV;
    cppTensorType* mb;
    cppTensorType* mc;

    cppTensorType* mFC_W;
    cppTensorType* mfc_b;

    // std::vector<cppTensorType*> mem;

    cppTensorType* dO;
    cppTensorType* dA;
    cppTensorType* dS;
    cppTensorType* dS_next;

    cppTensorTypeMap X;
    cppTensorTypeMap A;
    cppTensorTypeMap S;
    cppTensorTypeMap O;
};
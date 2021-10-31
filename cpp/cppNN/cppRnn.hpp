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
        this->dU_matMul = new cppTensor<dtype>(hidden_size, input_size);
        this->dW_matMul = new cppTensor<dtype>(hidden_size, hidden_size);
        this->FC_W_T = new cppTensor<dtype>(hidden_size, num_classes);
        this->O_T = new cppTensor<dtype>(1, hidden_size);
        this->S_T = new cppTensor<dtype>(1, hidden_size);
        this->V_T = new cppTensor<dtype>(hidden_size, hidden_size);
        this->X_T = new cppTensor<dtype>(1, input_size);
        this->W_T = new cppTensor<dtype>(hidden_size, hidden_size);

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
        SAFE_DELETE(dU_matMul)
        SAFE_DELETE(dW_matMul)
        SAFE_DELETE(FC_W_T);
        SAFE_DELETE(O_T)
        SAFE_DELETE(S_T)
        SAFE_DELETE(V_T)
        SAFE_DELETE(X_T)
        SAFE_DELETE(W_T)

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

    virtual void cuda_child() override
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
        dU_matMul->cuda();
        dW_matMul->cuda();
        FC_W_T->cuda();
        O_T->cuda();
        S_T->cuda();
        V_T->cuda();
        X_T->cuda();
        W_T->cuda();


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

    virtual void cpu_child() override
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
        dU_matMul->cpu();
        dW_matMul->cpu();
        FC_W_T->cpu();
        O_T->cpu();
        S_T->cpu();
        V_T->cpu();
        X_T->cpu();
        W_T->cpu();

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
    virtual void optimizer_gpu() override
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

    virtual void optimizer_cpu() override
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

    virtual void backward_gpu(cppTensor<dtype>& dY) override
    {
        dFC_W->zeros();
        dfc_b->zeros();
        dU->zeros();
        dW->zeros();
        dV->zeros();
        db->zeros();
        dc->zeros();
        dS_next->zeros();
        
        cppTensor_Functions::transpose_gpu(*O_T, *O[seq_length - 1]);
        cppTensor_Functions::matMul_gpu(*dFC_W, dY, *O_T, this->use_sharedMemory);
        cppTensor_Functions::copy_gpu(dfc_b, dY);

        cppTensor_Functions::transpose_gpu(*FC_W_T, *FC_W);
        cppTensor_Functions::matMul_gpu(*dO, *FC_W_T, dY, this->use_sharedMemory);

        cppTensor_Functions::transpose_gpu(*S_T, *S[seq_length - 1]);
        cppTensor_Functions::matMul_gpu(*dV, *dO, *S_T, this->use_sharedMemory);
        cppTensor_Functions::copy_gpu(dc, *dO);

        for (int t = seq_length - 1; t >= 0; --t)
        {
            cppTensor_Functions::transpose_gpu(*V_T, *V);
            cppTensor_Functions::matMul_gpu(*dS, *V_T, *dO, this->use_sharedMemory);
            cppTensor_Functions::add_gpu(*dS, *dS, *dS_next);
            cppTensor_Functions::deriv_tanh_gpu(dA, *S[t]);
            cppTensor_Functions::mul_gpu(dA, *dA, *dS);
            cppTensor_Functions::transpose_gpu(*X_T, *X[t]);
            cppTensor_Functions::matMul_gpu(*dU_matMul, *dA, *X_T, this->use_sharedMemory);
            cppTensor_Functions::add_gpu(*dU, *dU, *dU_matMul);
            cppTensor_Functions::transpose_gpu(*S_T, *S[t - 1]);
            cppTensor_Functions::matMul_gpu(*dW_matMul, *dA, *S_T, this->use_sharedMemory);
            cppTensor_Functions::add_gpu(*dW, *dW, *dW_matMul);
            cppTensor_Functions::add_gpu(*db, *db, *dA);
            cppTensor_Functions::transpose_gpu(*W_T, *W);
            cppTensor_Functions::matMul_gpu(*dS_next, *W_T, *dA, this->use_sharedMemory);
        }
    }

    virtual void backward_cpu(cppTensor<dtype>& dY) override
    {
        dFC_W->zeros();
        dfc_b->zeros();
        dU->zeros();
        dW->zeros();
        dV->zeros();
        db->zeros();
        dc->zeros();
        dS_next->zeros();

        cppTensor_Functions::transpose_cpu(O_T, *O[seq_length - 1]);
        cppTensor_Functions::matMul_cpu(dFC_W, dY, *O_T);
        cppTensor_Functions::copy_cpu(dfc_b, dY);

        cppTensor_Functions::transpose_cpu(FC_W_T, *FC_W);
        cppTensor_Functions::matMul_cpu(dO, *FC_W_T, dY);

        cppTensor_Functions::transpose_cpu(S_T, *S[seq_length - 1]);
        cppTensor_Functions::matMul_cpu(dV, *dO, *S_T);
        cppTensor_Functions::copy_cpu(dc, *dO);

        for (int t = seq_length - 1; t >= 0; --t)
        {
            cppTensor_Functions::transpose_cpu(V_T, *V);
            cppTensor_Functions::matMul_cpu(dS, *V_T, *dO);
            cppTensor_Functions::add_cpu(dS, *dS, *dS_next);
            cppTensor_Functions::deriv_tanh_cpu(dA, *S[t]);
            cppTensor_Functions::mul_cpu(dA, *dA, *dS);
            cppTensor_Functions::transpose_cpu(X_T, *X[t]);
            cppTensor_Functions::matMul_cpu(dU_matMul, *dA, *X_T);
            cppTensor_Functions::add_cpu(dU, *dU, *dU_matMul);
            cppTensor_Functions::transpose_cpu(S_T, *S[t - 1]);
            cppTensor_Functions::matMul_cpu(dW_matMul, *dA, *S_T);
            cppTensor_Functions::add_cpu(dW, *dW, *dW_matMul);
            cppTensor_Functions::add_cpu(db, *db, *dA);
            cppTensor_Functions::transpose_cpu(W_T, *W);
            cppTensor_Functions::matMul_cpu(dS_next, *W_T, *dA);
        }
    }

    virtual void cross_entropy_loss_gpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) override
    {
        cppTensor_Functions::softmax_gpu(&dY, outputs);
        cppTensor_Functions::copy_gpu(&Y, dY);

        cppTensor_Functions::log_gpu(&Y);
        cppTensor_Functions::minus_gpu(&Y);
        cppTensor_Functions::deriv_softmax_gpu(dY, loss, Y, labels);
    }

    virtual void cross_entropy_loss_cpu(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) override
    {
        cppTensor_Functions::softmax_cpu(&dY, outputs);
        cppTensor_Functions::copy_cpu(&Y, dY);
        
        cppTensor_Functions::log_cpu(&Y);
        cppTensor_Functions::minus_cpu(&Y);
        cppTensor_Functions::deriv_softmax_cpu(dY, loss, Y, labels);
    }

    virtual void forward_gpu(cppTensor<dtype>& outputs, const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev) override
    {
        cppTensor_Functions::copy_gpu(S[-1], hprev);

        for (int t = 0; t < seq_length; ++t)
        {
            cppTensor_Functions::transpose_gpu(*X[t], x[t]);
            cppTensor_Functions::add_gpu(*A[t], matMul(*U, *X[t], this->use_sharedMemory), matMul(*W, *S[t - 1], this->use_sharedMemory));
            cppTensor_Functions::add_gpu(*A[t], *A[t], *b);
            cppTensor_Functions::tanh_gpu(S[t], *A[t]);
            cppTensor_Functions::matMul_gpu(*O[seq_length], *V, *S[t], this->use_sharedMemory);
            cppTensor_Functions::add_gpu(*O[t], *O[seq_length], *c);
        }

        cppTensor_Functions::matMul_gpu(outputs, *FC_W, *O[seq_length-1], this->use_sharedMemory);
        cppTensor_Functions::add_gpu(outputs, outputs, *fc_b);
    }

    virtual void forward_cpu(cppTensor<dtype>& outputs, const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev) override
    {
        cppTensor_Functions::copy_cpu(S[-1], hprev);
        
        for (int t = 0; t < seq_length; ++t)
        {
            cppTensor_Functions::transpose_cpu(X[t], x[t]);
            cppTensor_Functions::add_cpu(A[t], matMul(*U, *X[t], this->use_sharedMemory), matMul(*W, *S[t - 1], this->use_sharedMemory));
            cppTensor_Functions::add_cpu(A[t], *A[t], *b);
            cppTensor_Functions::tanh_cpu(S[t], *A[t]);
            cppTensor_Functions::matMul_cpu(O[seq_length], *V, *S[t]);
            cppTensor_Functions::add_cpu(O[t], *O[seq_length], *c);
        }

        cppTensor_Functions::matMul_cpu(&outputs, *FC_W, *O[seq_length-1]);
        cppTensor_Functions::add_cpu(&outputs, outputs, *fc_b);
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
    cppTensorType* dU_matMul;
    cppTensorType* dW_matMul;
    cppTensorType* FC_W_T;
    cppTensorType* O_T;
    cppTensorType* S_T;
    cppTensorType* V_T;
    cppTensorType* X_T;
    cppTensorType* W_T;
    

    cppTensorTypeMap X;
    cppTensorTypeMap A;
    cppTensorTypeMap S;
    cppTensorTypeMap O;
};
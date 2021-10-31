#pragma once

#include "cppNN.hpp"

template<typename dtype>
class cppRnn : public cppNN<dtype>
{
public:
    using numpyArray = pybind11::array_t<dtype, pybind11::array::c_style>;
    using numpyArrayGeneric = pybind11::array;
    using cppTensorType = cppTensor<dtype>;
    using mapIntCppTensor = std::unordered_map<int, cppTensor<dtype>*>;
    using mapStrCppTensor = std::unordered_map<std::string, cppTensor<dtype>*>;
    using mapIntCppTensorIter = std::unordered_map<int, cppTensor<double>*>::iterator;
    using mapStrCppTensorIter = std::unordered_map<std::string, cppTensor<double>*>::iterator;

public:
    cppRnn(double lr, const cppTensorType& U, const cppTensorType& W, const cppTensorType& V, const cppTensorType& FC_W,
        int seq_length, int input_size, int hidden_size, int num_classes)
    {
        this->lr = lr;
        this->seq_length = seq_length;
        this->hidden_size = hidden_size;
        
        this->params["U"] = new cppTensor<dtype>(hidden_size, input_size);
        this->params["W"] = new cppTensor<dtype>(hidden_size, hidden_size);
        this->params["V"] = new cppTensor<dtype>(hidden_size, hidden_size);
        this->params["b"] = new cppTensor<dtype>(hidden_size, 1);
        this->params["c"] = new cppTensor<dtype>(hidden_size, 1);

        this->params["FC_W"] = new cppTensor<dtype>(num_classes, hidden_size);
        this->params["fc_b"] = new cppTensor<dtype>(num_classes, 1);

        copy(this->params["U"], U);
        copy(this->params["W"], W);
        copy(this->params["V"], V);

        copy(this->params["FC_W"], FC_W);

        this->dparams["dU"] = new cppTensor<dtype>(hidden_size, input_size);
        this->dparams["dW"] = new cppTensor<dtype>(hidden_size, hidden_size);
        this->dparams["dV"] = new cppTensor<dtype>(hidden_size, hidden_size);
        this->dparams["db"] = new cppTensor<dtype>(hidden_size, 1);
        this->dparams["dc"] = new cppTensor<dtype>(hidden_size, 1);

        this->dparams["dFC_W"] = new cppTensor<dtype>(num_classes, hidden_size);
        this->dparams["dfc_b"] = new cppTensor<dtype>(num_classes, 1);

        this->mem["mU"] = new cppTensor<dtype>(hidden_size, input_size);
        this->mem["mW"] = new cppTensor<dtype>(hidden_size, hidden_size);
        this->mem["mV"] = new cppTensor<dtype>(hidden_size, hidden_size);
        this->mem["mb"] = new cppTensor<dtype>(hidden_size, 1);
        this->mem["mc"] = new cppTensor<dtype>(hidden_size, 1);

        this->mem["mFC_W"] = new cppTensor<dtype>(num_classes, hidden_size);
        this->mem["mfc_b"] = new cppTensor<dtype>(num_classes, 1);
        
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
        SAFE_DELETE(dO)
        SAFE_DELETE(dA)
        SAFE_DELETE(dS)
        SAFE_DELETE(dS_next)

        // params.clear();
        // dparams.clear();
        // mem.clear();

        mapIntCppTensorIter mapIntIter;
        mapStrCppTensorIter mapStrIter;

        for (mapStrIter = this->params.begin(); mapStrIter != this->params.end(); ++mapStrIter)
        {
            SAFE_DELETE(mapStrIter->second)
        }

        for (mapStrIter = this->dparams.begin(); mapStrIter != this->dparams.end(); ++mapStrIter)
        {
            SAFE_DELETE(mapStrIter->second)
        }

        for (mapStrIter = this->mem.begin(); mapStrIter != this->mem.end(); ++mapStrIter)
        {
            SAFE_DELETE(mapStrIter->second)
        }

        for (mapIntIter = this->S.begin(); mapIntIter != this->S.end(); ++mapIntIter)
        {
            SAFE_DELETE(mapIntIter->second)
        }

        for (mapIntIter = this->X.begin(); mapIntIter != this->X.end(); ++mapIntIter)
        {
            SAFE_DELETE(mapIntIter->second)
        }

        for (mapIntIter = this->A.begin(); mapIntIter != this->A.end(); ++mapIntIter)
        {
            SAFE_DELETE(mapIntIter->second)
        }

        for (mapIntIter = this->O.begin(); mapIntIter != this->O.end(); ++mapIntIter)
        {
            SAFE_DELETE(mapIntIter->second)
        }
    }

    virtual void cuda_impl() override
    {

        dO->cuda();
        dA->cuda();
        dS->cuda();
        dS_next->cuda();

        mapIntCppTensorIter mapIntIter;
        mapStrCppTensorIter mapStrIter;

        for (mapStrIter = this->params.begin(); mapStrIter != this->params.end(); ++mapStrIter)
        {
            if (mapStrIter->second)
            {
                mapStrIter->second->cuda();
            }
        }

        for (mapStrIter = this->dparams.begin(); mapStrIter != this->dparams.end(); ++mapStrIter)
        {
            if (mapStrIter->second)
            {
                mapStrIter->second->cuda();
            }
        }

        for (mapStrIter = this->mem.begin(); mapStrIter != this->mem.end(); ++mapStrIter)
        {
            if (mapStrIter->second)
            {
                mapStrIter->second->cuda();
            }
        }

        for (mapIntIter = this->S.begin(); mapIntIter != this->S.end(); ++mapIntIter)
        {
            if (mapIntIter->second)
            {
                mapIntIter->second->cuda();
            }
        }

        for (mapIntIter = this->X.begin(); mapIntIter != this->X.end(); ++mapIntIter)
        {
            if (mapIntIter->second)
            {
                mapIntIter->second->cuda();
            }
        }

        for (mapIntIter = this->A.begin(); mapIntIter != this->A.end(); ++mapIntIter)
        {
            if (mapIntIter->second)
            {
                mapIntIter->second->cuda();
            }
        }

        for (mapIntIter = this->O.begin(); mapIntIter != this->O.end(); ++mapIntIter)
        {
            if (mapIntIter->second)
            {
                mapIntIter->second->cuda();
            }
        }
    }

    virtual void cpu_impl() override
    {
        dO->cpu();
        dA->cpu();
        dS->cpu();
        dS_next->cpu();

        mapIntCppTensorIter mapIntIter;
        mapStrCppTensorIter mapStrIter;

        for (mapStrIter = this->params.begin(); mapStrIter != this->params.end(); ++mapStrIter)
        {
            if (mapStrIter->second)
            {
                mapStrIter->second->cpu();
            }
        }

        for (mapStrIter = this->dparams.begin(); mapStrIter != this->dparams.end(); ++mapStrIter)
        {
            if (mapStrIter->second)
            {
                mapStrIter->second->cpu();
            }
        }

        for (mapStrIter = this->mem.begin(); mapStrIter != this->mem.end(); ++mapStrIter)
        {
            if (mapStrIter->second)
            {
                mapStrIter->second->cpu();
            }
        }

        for (mapIntIter = this->S.begin(); mapIntIter != this->S.end(); ++mapIntIter)
        {
            if (mapIntIter->second)
            {
                mapIntIter->second->cpu();
            }
        }

        for (mapIntIter = this->X.begin(); mapIntIter != this->X.end(); ++mapIntIter)
        {
            if (mapIntIter->second)
            {
                mapIntIter->second->cpu();
            }
        }

        for (mapIntIter = this->A.begin(); mapIntIter != this->A.end(); ++mapIntIter)
        {
            if (mapIntIter->second)
            {
                mapIntIter->second->cpu();
            }
        }

        for (mapIntIter = this->O.begin(); mapIntIter != this->O.end(); ++mapIntIter)
        {
            if (mapIntIter->second)
            {
                mapIntIter->second->cpu();
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
        clip(this->dparams["dU"], -5.0, 5.0);
        clip(this->dparams["dW"], -5.0, 5.0);
        clip(this->dparams["dV"], -5.0, 5.0);
        clip(this->dparams["db"], -5.0, 5.0);
        clip(this->dparams["dc"], -5.0, 5.0);
        clip(this->dparams["dFC_W"], -5.0, 5.0);
        clip(this->dparams["dfc_b"], -5.0, 5.0);
        
        optimizer(this->params["U"], this->mem["mU"], *this->dparams["dU"], -lr);
        optimizer(this->params["W"], this->mem["mW"], *this->dparams["dW"], -lr);
        optimizer(this->params["V"], this->mem["mV"], *this->dparams["dV"], -lr);
        optimizer(this->params["b"], this->mem["mb"], *this->dparams["db"], -lr);
        optimizer(this->params["c"], this->mem["mc"], *this->dparams["dc"], -lr);
        optimizer(this->params["FC_W"], this->mem["mFC_W"], *this->dparams["dFC_W"], -lr);
        optimizer(this->params["fc_b"], this->mem["mfc_b"], *this->dparams["dfc_b"], -lr);
    }

    virtual std::vector<cppTensor<dtype>> backward_impl(const cppTensor<dtype>& dY) override
    {
        this->dparams["dFC_W"]->zeros();
        this->dparams["dfc_b"]->zeros();
        this->dparams["dU"]->zeros();
        this->dparams["dW"]->zeros();
        this->dparams["dV"]->zeros();
        this->dparams["db"]->zeros();
        this->dparams["dc"]->zeros();

        dS_next->zeros();
        
        *this->dparams["dFC_W"] = matMul(dY, transpose(*O[seq_length - 1]), this->use_sharedMemory);
        copy(this->dparams["dfc_b"] , dY);

        *dO = matMul(transpose(*this->params["FC_W"]), dY, this->use_sharedMemory);

        *this->dparams["dV"] = matMul(*dO, transpose(*S[seq_length - 1]), this->use_sharedMemory);
        copy(this->dparams["dc"], *dO);

        for (int t = seq_length - 1; t >= 0; --t)
        {
            *dS = matMul(transpose(*this->params["V"]), *dO, this->use_sharedMemory) + *dS_next;
            *dA = deriv_tanh(*S[t]) * (*dS);
            
            *this->dparams["dU"] = *this->dparams["dU"] + matMul(*dA, transpose(*X[t]), this->use_sharedMemory);
            *this->dparams["dW"] = *this->dparams["dW"] + matMul(*dA, transpose(*S[t - 1]), this->use_sharedMemory);
            *this->dparams["db"] = *this->dparams["db"] + *dA;
            *dS_next = matMul(transpose(*this->params["W"]), *dA, this->use_sharedMemory);
        }

        return std::vector<cppTensor<dtype>>(
            {*this->dparams["dU"], *this->dparams["dW"], *this->dparams["dV"], 
            *this->dparams["db"], *this->dparams["dc"], *this->dparams["dFC_W"], *this->dparams["dfc_b"]});
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
            *A[t] = matMul(*this->params["U"], *X[t], this->use_sharedMemory) + matMul(*this->params["W"], *S[t - 1], this->use_sharedMemory) + *this->params["b"];
            *S[t] = tanh(*A[t]);
            *O[t] = matMul(*this->params["V"], *S[t], this->use_sharedMemory) + *this->params["c"];
        }

        return matMul(*this->params["FC_W"], *O[seq_length-1], this->use_sharedMemory) + *this->params["fc_b"];
    }

private:
    double lr{0.0};
    size_t seq_length{0};
    size_t hidden_size{0};

    cppTensorType* dO;
    cppTensorType* dA;
    cppTensorType* dS;
    cppTensorType* dS_next;

    mapStrCppTensor params;
    mapStrCppTensor dparams;
    mapStrCppTensor mem;

    mapIntCppTensor X;
    mapIntCppTensor A;
    mapIntCppTensor S;
    mapIntCppTensor O;
};
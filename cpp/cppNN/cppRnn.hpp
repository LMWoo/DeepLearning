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

        this->rParams["U"] = new cppTensor<dtype>(hidden_size, input_size);
        this->rParams["W"] = new cppTensor<dtype>(hidden_size, hidden_size);
        this->rParams["V"] = new cppTensor<dtype>(hidden_size, hidden_size);
        this->rParams["b"] = new cppTensor<dtype>(hidden_size, 1);
        this->rParams["c"] = new cppTensor<dtype>(hidden_size, 1);

        this->rParams["FC_W"] = new cppTensor<dtype>(num_classes, hidden_size);
        this->rParams["fc_b"] = new cppTensor<dtype>(num_classes, 1);

        copy(this->params["U"], U);
        copy(this->params["W"], W);
        copy(this->params["V"], V);

        copy(this->params["FC_W"], FC_W);

        copy(this->rParams["U"], U);
        copy(this->rParams["W"], W);
        copy(this->rParams["V"], V);

        copy(this->rParams["FC_W"], FC_W);

        this->params["dU"]    = new cppTensor<dtype>(hidden_size, input_size);
        this->params["dW"]    = new cppTensor<dtype>(hidden_size, hidden_size);
        this->params["dV"]    = new cppTensor<dtype>(hidden_size, hidden_size);
        this->params["db"]    = new cppTensor<dtype>(hidden_size, 1);
        this->params["dc"]    = new cppTensor<dtype>(hidden_size, 1);
        this->params["dFC_W"] = new cppTensor<dtype>(num_classes, hidden_size);
        this->params["dfc_b"] = new cppTensor<dtype>(num_classes, 1);

        this->rParams["dU"]    = new cppTensor<dtype>(hidden_size, input_size);
        this->rParams["dW"]    = new cppTensor<dtype>(hidden_size, hidden_size);
        this->rParams["dV"]    = new cppTensor<dtype>(hidden_size, hidden_size);
        this->rParams["db"]    = new cppTensor<dtype>(hidden_size, 1);
        this->rParams["dc"]    = new cppTensor<dtype>(hidden_size, 1);
        this->rParams["dFC_W"] = new cppTensor<dtype>(num_classes, hidden_size);
        this->rParams["dfc_b"] = new cppTensor<dtype>(num_classes, 1);

        this->mem["mU"]    = new cppTensor<dtype>(hidden_size, input_size);
        this->mem["mW"]    = new cppTensor<dtype>(hidden_size, hidden_size);
        this->mem["mV"]    = new cppTensor<dtype>(hidden_size, hidden_size);
        this->mem["mb"]    = new cppTensor<dtype>(hidden_size, 1);
        this->mem["mc"]    = new cppTensor<dtype>(hidden_size, 1);
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
        SAFE_DELETE(dO)
        SAFE_DELETE(dA)
        SAFE_DELETE(dS)
        SAFE_DELETE(dS_next)

        mapIntCppTensorIter mapIntIter;
        mapStrCppTensorIter mapStrIter;

        for (mapStrIter = this->params.begin(); mapStrIter != this->params.end(); ++mapStrIter)
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

        for (mapStrIter = this->rParams.begin(); mapStrIter != this->rParams.end(); ++mapStrIter)
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


        for (mapStrIter = this->rParams.begin(); mapStrIter != this->rParams.end(); ++mapStrIter)
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
    virtual mapStrCppTensor parameters_impl() override
    {
        for (auto iter = this->params.begin(); iter != this->params.end(); ++iter)
        {
            this->rParams[iter->first]->shape_.rows = this->params[iter->first]->shape_.rows;
            this->rParams[iter->first]->shape_.cols = this->params[iter->first]->shape_.cols;

            if (this->is_cuda_)
            {
                this->rParams[iter->first]->dev_data_ = this->params[iter->first]->dev_data_;
            }
            else
            {
                this->rParams[iter->first]->data_ = this->params[iter->first]->data_;
            }
            
            this->rParams[iter->first]->is_owner_ = false;
        }

        return this->rParams;
    }


    virtual cppTensor<dtype> forward_impl(const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev) override
    {
        copy(S[-1], hprev);

        for (int t = 0; t < seq_length; ++t)
        {
            *X[t] = transpose(x[t]);
            *A[t] = transpose_matMul(*this->params["U"], x[t]) + matMul(*this->params["W"], *S[t - 1], this->use_sharedMemory) + *this->params["b"];
            *S[t] = tanh(*A[t]);
            *O[t] = matMul(*this->params["V"], *S[t], this->use_sharedMemory) + *this->params["c"];
        }

        return matMul(*this->params["FC_W"], *O[seq_length-1], this->use_sharedMemory) + *this->params["fc_b"];
    }

    virtual void backward_impl(const cppTensor<dtype>& dY) override
    {
        dS_next->zeros();

        copy(this->params["dFC_W"], transpose_matMul(dY, *O[seq_length - 1]));
        copy(this->params["dfc_b"], dY);

        *dO = matMul(transpose(*this->params["FC_W"]), dY, this->use_sharedMemory);

        copy(this->params["dV"], transpose_matMul(*dO, *S[seq_length - 1]));
        copy(this->params["dc"], *dO);

        for (int t = seq_length - 1; t >= 0; --t)
        {
            *dS = matMul(transpose(*this->params["V"]), *dO, this->use_sharedMemory) + *dS_next;
            *dA = deriv_tanh(*S[t]) * (*dS);
            
            copy(this->params["dU"], *this->params["dU"] + transpose_matMul(*dA, *X[t]));
            copy(this->params["dW"], *this->params["dW"] + transpose_matMul(*dA, *S[t - 1]));
            copy(this->params["db"], *this->params["db"] + *dA);

            *dS_next = matMul(transpose(*this->params["W"]), *dA, this->use_sharedMemory);
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
private:
    double lr{0.0};
    size_t seq_length{0};
    size_t hidden_size{0};

    cppTensorType* dO;
    cppTensorType* dA;
    cppTensorType* dS;
    cppTensorType* dS_next;

    mapStrCppTensor params;
    mapStrCppTensor rParams;
    mapStrCppTensor mem;

    mapIntCppTensor X;
    mapIntCppTensor A;
    mapIntCppTensor S;
    mapIntCppTensor O;
};
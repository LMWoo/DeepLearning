#pragma once

#include <NumCpp.hpp>
#include <NumCpp/Core/Types.hpp>
#include <NumCpp/Core/Shape.hpp>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace nc;

namespace nnCpp
{
    template<typename dtype>
    NdArray<dtype> xavier_init(uint32 c1, uint32 c2, uint32 w = 1, uint32 h = 1)
    {
        double fan_1 = c2 * w * h;
        double fan_2 = c1 * w * h;
        double ratio = std::sqrt(6.0 / (fan_1 + fan_2));
        NdArray<dtype> params = ratio * (2.0 * random::rand<dtype>(0.0, 1.0, Shape(c1, c2)) - 1.0);
        return params;
    }

    template<typename dtype>
    class rnn
    {
    private:
        using array_list = std::unordered_map<int, NdArray<dtype>>;
        using pbArray = pybind11::array_t<dtype, pybind11::array::c_style>;
        using pbArrayGeneric = pybind11::array;

    public:
        rnn(double lr, uint32 seq_length, uint32 input_size, uint32 hidden_size, uint32 num_classes)
        {
#ifdef RNN_DEBUG
            this->lr = lr;
            this->seq_length = seq_length;
            this->hidden_size = hidden_size;
            this->U = xavier_init<dtype>(hidden_size, input_size);
            this->W = xavier_init<dtype>(hidden_size, hidden_size);
            this->V = xavier_init<dtype>(hidden_size, hidden_size);
            this->b = zeros<dtype>(hidden_size, 1);
            this->c = zeros<dtype>(hidden_size, 1);

            this->FC_W = xavier_init<dtype>(num_classes, hidden_size);
            this->fc_b = zeros<dtype>(num_classes, 1);
            
            this->mU = zeros_like<dtype, dtype>(this->U);
            this->mW = zeros_like<dtype, dtype>(this->W);
            this->mV = zeros_like<dtype, dtype>(this->V);
            this->mb = zeros_like<dtype, dtype>(this->b);
            this->mc = zeros_like<dtype, dtype>(this->c);

            this->mFC_W = zeros_like<dtype, dtype>(this->FC_W);
            this->mfc_b = zeros_like<dtype, dtype>(this->fc_b);
#else
            this->lr = lr;
            this->seq_length = seq_length;
            this->hidden_size = hidden_size;
            this->U = xavier_init<dtype>(hidden_size, input_size);
            this->W = xavier_init<dtype>(hidden_size, hidden_size);
            this->V = xavier_init<dtype>(hidden_size, hidden_size);
            this->b = zeros<dtype>(hidden_size, 1);
            this->c = zeros<dtype>(hidden_size, 1);

            this->FC_W = xavier_init<dtype>(num_classes, hidden_size);
            this->fc_b = zeros<dtype>(num_classes, 1);
            
            this->mU = zeros_like<dtype, dtype>(this->U);
            this->mW = zeros_like<dtype, dtype>(this->W);
            this->mV = zeros_like<dtype, dtype>(this->V);
            this->mb = zeros_like<dtype, dtype>(this->b);
            this->mc = zeros_like<dtype, dtype>(this->c);

            this->mFC_W = zeros_like<dtype, dtype>(this->FC_W);
            this->mfc_b = zeros_like<dtype, dtype>(this->fc_b);
#endif
        }
        
        NdArray<dtype> forward(pbArray x, pbArray hprev)
        {
            rnn_debug_start = true;
#ifdef RNN_DEBUG
            NdArray<dtype> returnArray(1, 1);
            returnArray.autoMemoryFreeOff();
            return returnArray;
#else
            std::vector<NdArray<dtype>> returnVec;
            NdArray<dtype> returnHprev = NdArray<dtype>(hprev);
            
            auto dataPtr = x.data();  
            const uint32 numRows = static_cast<uint32>(x.shape(1));
            const uint32 numCols = static_cast<uint32>(x.shape(2));
            const uint32 image_size = numRows * numCols;

            for (int i = 0; i < x.shape(0); ++i)
            {
                NdArray<dtype> newArray(numRows, numCols);
                newArray.autoMemoryFreeOff();
                std::copy(dataPtr + image_size * i, dataPtr + image_size * (i + 1), newArray.begin());

                returnVec.push_back(newArray);
            }

            returnHprev.autoMemoryFreeOff();
            return forward_(returnVec, returnHprev);
#endif
        }

        NdArray<dtype> forward_(std::vector<NdArray<dtype>> x, NdArray<dtype> hprev)
        {
#ifdef RNN_DEBUG
            return NdArray<dtype>(1, 1);
#else
            S[-1] = NdArray<dtype>(hprev);

            for (int t = 0; t < (int)seq_length; ++t)
            {
                X[t] = x[t].transpose();
                
                // printf("--------------shape print -----------\n");
                // U.shape().print();
                // X[t].shape().print();
                // W.shape().print();
                // S[t - 1].shape().print();
                // b.shape().print();
                // printf("-------------------------------------\n");

                A[t] = dot<dtype>(U, X[t]) + dot<dtype>(W, S[t - 1]) + b;
                S[t] = tanh<dtype>(A[t]);
                O[t] = dot<dtype>(V, S[t]) + c;
            }

            O[(int)seq_length - 1].shape().print();
            FC_O = dot<dtype>(FC_W, O[(int)seq_length - 1]) + fc_b;
            FC_O.autoMemoryFreeOff();

            for (int i = 0; i < x.size(); ++i)
            {
                x[i].memoryFree();
            }
            hprev.memoryFree();
            return FC_O;
#endif
        }

        auto backward(const NdArray<dtype>& dY)
        {
#ifdef RNN_DEBUG
            return std::make_tuple(
                NdArray<dtype>(1, 1), 
                NdArray<dtype>(1, 1), 
                NdArray<dtype>(1, 1), 
                NdArray<dtype>(1, 1), 
                NdArray<dtype>(1, 1), 
                NdArray<dtype>(1, 1), 
                NdArray<dtype>(1, 1));
#else
            NdArray<dtype> dFC_W = zeros_like<dtype, dtype>(FC_W);
            NdArray<dtype> dfc_b = zeros_like<dtype, dtype>(fc_b);

            NdArray<dtype> dU = zeros_like<dtype, dtype>(U);
            NdArray<dtype> dW = zeros_like<dtype, dtype>(W);
            NdArray<dtype> dV = zeros_like<dtype, dtype>(V);

            NdArray<dtype> db = zeros_like<dtype, dtype>(b);
            NdArray<dtype> dc = zeros_like<dtype, dtype>(c);
            NdArray<dtype> dS_next = zeros_like<dtype, dtype>(S[0]);

            dFC_W = dot<dtype>(dY, O[seq_length - 1].transpose());
            dfc_b = dY;
            NdArray<dtype> dO = dot<dtype>(FC_W.transpose(), dY);

            dV = dot<dtype>(dO, S[seq_length - 1].transpose());
            dc = dO;

            for (int t = seq_length - 1; t >= 0; --t)
            {
                NdArray<dtype> dS = dot<dtype>(V.transpose(), dO) + dS_next;
                NdArray<dtype> dA = (1.0 - S[t] * S[t]) * dS;
                dU = dU + dot<dtype>(dA, X[t].transpose());
                NdArray<dtype> temp = dot<dtype>(dA, S[t - 1].transpose());
                dW = dW + dot<dtype>(dA, S[t - 1].transpose());
                db = db + dA;
                dS_next = dot<dtype>(W.transpose(), dA);
            }

            return std::make_tuple(dU, dW, dV, db, dc, dFC_W, dfc_b);
#endif
        }

        void optimizer(std::tuple<NdArray<dtype>, NdArray<dtype>, NdArray<dtype>, NdArray<dtype>, NdArray<dtype>, NdArray<dtype>, NdArray<dtype>> gradients)
        {

#ifdef RNN_DEBUG

#else
            // auto param = std::make_tuple(U, W, V, b, c, FC_W, fc_b);
            // auto mem = std::make_tuple(mU, mW, mV, mb, mc, mFC_W, mfc_b);

            // for (int i = 0; i < 7; ++i)
            // {
            //     NdArray<dtype> m = std::get<i>(mem);
            //     NdArray<dtype> p = std::get<i>(param);
            //     NdArray<dtype> g = std::get<i>(gradients);
            //     m = m + g * g;
            //     p = p + (-lr) * p / sqrt<dtype>(m + 1e-8);
            // }

            NdArray<dtype> dU = std::get<0>(gradients);

            mU = mU + dU * dU;
            U = U + (-lr) * U / sqrt<dtype>(mU + 1e-8);


            NdArray<dtype> dW = std::get<1>(gradients);

            mW = mW + dW * dW;
            W = W + (-lr) * W / sqrt<dtype>(mW + 1e-8);


            NdArray<dtype> dV = std::get<2>(gradients);

            mV = mV + dV * dV;
            V = V + (-lr) * V / sqrt<dtype>(mV + 1e-8);


            NdArray<dtype> db = std::get<3>(gradients);

            mb = mb + db * db;
            b = b + (-lr) * b / sqrt<dtype>(mb + 1e-8);


            NdArray<dtype> dc = std::get<4>(gradients);

            mc = mc + dc * dc;
            c = c + (-lr) * c / sqrt<dtype>(mc + 1e-8);


            NdArray<dtype> dFC_W = std::get<5>(gradients);

            mFC_W = mFC_W + dFC_W * dFC_W;
            FC_W = FC_W + (-lr) * FC_W / sqrt<dtype>(mFC_W + 1e-8);


            NdArray<dtype> dfc_b = std::get<6>(gradients);

            mfc_b = mfc_b + dfc_b * dfc_b;
            fc_b = fc_b + (-lr) * fc_b / sqrt<dtype>(mfc_b + 1e-8);

#endif
        }

        auto cross_entropy_loss(NdArray<dtype> outputs, NdArray<dtype> labels)
        {
            NdArray<dtype> Y = softmax(outputs);
            NdArray<dtype> loss = -log<dtype>(Y) * one_hot_vector(Y, labels);

            return std::make_tuple(Y, loss);
        }

        NdArray<dtype> softmax(const NdArray<dtype>& x)
        {
            NdArray<dtype> e = exp<dtype>(x);
            return e / *sum<dtype>(e).begin();
        }

        NdArray<dtype> deriv_softmax(NdArray<dtype> Y, NdArray<dtype> labels)
        {
            NdArray<dtype> dY = Y;
            for (int i = 0; i < labels.shape().cols; ++i)
            {
                dY((int)labels(0, i), i) -= 1.0;
            }
            return dY;
        }

        NdArray<dtype> one_hot_vector(NdArray<dtype> Y, NdArray<dtype> labels)
        {
            NdArray<dtype> out = zeros_like<dtype>(Y);
            for (int i = 0; i < labels.shape().cols; ++i)
            {
                out((int)labels(0, i), i) = 1.0;
            }

            return out;
        }
        
        void test()
        {
            printf("rnn_test\n");
        }

    private:
        double lr{0.0};
        uint32 seq_length{0};
        uint32 hidden_size{0};

    private:

#ifdef RNN_DEBUG
        NdArray<dtype> U{NdArray<dtype>(1, 1)};
        NdArray<dtype> W{NdArray<dtype>(1, 1)};
        NdArray<dtype> V{NdArray<dtype>(1, 1)};
        NdArray<dtype> b{NdArray<dtype>(1, 1)};
        NdArray<dtype> c{NdArray<dtype>(1, 1)};

        NdArray<dtype> FC_W{NdArray<dtype>(1, 1)};
        NdArray<dtype> fc_b{NdArray<dtype>(1, 1)};

        NdArray<dtype> mU{NdArray<dtype>(1, 1)};
        NdArray<dtype> mW{NdArray<dtype>(1, 1)};
        NdArray<dtype> mV{NdArray<dtype>(1, 1)};
        NdArray<dtype> mb{NdArray<dtype>(1, 1)};
        NdArray<dtype> mc{NdArray<dtype>(1, 1)};

        NdArray<dtype> mFC_W{NdArray<dtype>(1, 1)};
        NdArray<dtype> mfc_b{NdArray<dtype>(1, 1)};

        array_list X;
        array_list A;
        array_list S;
        array_list O;
        NdArray<dtype> FC_O{NdArray<dtype>(1, 1)};
#else
        NdArray<dtype> U;
        NdArray<dtype> W;
        NdArray<dtype> V;
        NdArray<dtype> b;
        NdArray<dtype> c;

        NdArray<dtype> FC_W;
        NdArray<dtype> fc_b;

        NdArray<dtype> mU;
        NdArray<dtype> mW;
        NdArray<dtype> mV;
        NdArray<dtype> mb;
        NdArray<dtype> mc;

        NdArray<dtype> mFC_W;
        NdArray<dtype> mfc_b;

        array_list X;
        array_list A;
        array_list S;
        array_list O;
        NdArray<dtype> FC_O;
#endif
    };
}
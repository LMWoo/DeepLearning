// #pragma once

// #include "cppNN.hpp"

// template<typename dtype>
// class cppSoftmax : public cppNN<dtype>
// {
// public:
//     cppSoftmax()
//     {
//     }

//     ~cppSoftmax()
//     {
//     }
// protected:
//     virtual void cuda_impl() override
//     {
//     }

//     virtual void cpu_impl() override
//     {
//     }

//     virtual void optimizer_impl() override
//     {
//     }

//     virtual std::vector<cppTensor<dtype>> backward_impl(const cppTensor<dtype>& dY) override
//     {
//         return std::vector<cppTensor<dtype>>({cppTensor<dtype>()});
//     }

//     virtual void cross_entropy_loss_impl(cppTensor<dtype>& dY, cppTensor<dtype>& Y, cppTensor<dtype>& loss, const cppTensor<dtype>& outputs, const cppTensor<dtype>& labels) override
//     {
//     }

//     virtual cppTensor<dtype> forward_impl(const std::vector<cppTensor<dtype>>& x, const cppTensor<dtype>& hprev) override
//     {
//         return cppTensor<dtype>();
//     }
// };
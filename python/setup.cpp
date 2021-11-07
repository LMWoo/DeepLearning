#include <torch/extension.h>
#include <cppTensor/cppTensor.hpp>
#include <cppModules/cppRnn.hpp>
#include <cppModules/cppLSTM.hpp>
#include <cppModules/cppGRU.hpp>
#include <cppOptimizer/cppAdagrad.hpp>
#include <cppLoss/cppCrossEntropyLoss.hpp>

namespace pb11 = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using cppTensorDouble = cppTensor<double>;
    pb11::class_<cppTensorDouble>(m, "cppTensor")
        .def(pb11::init<>())
        .def(pb11::init<size_t, size_t>())
        .def(pb11::init<size_t, size_t, bool>())
        .def(pb11::init<cppTensorDouble::numpyArray>())
        .def("cuda", &cppTensorDouble::cuda)
        .def("cpu", &cppTensorDouble::cpu)
        .def("numpy", &cppTensorDouble::numpy)
        .def("zeros", &cppTensorDouble::zeros)
        .def("ones", &cppTensorDouble::ones)
        .def("print_pointer", &cppTensorDouble::print_pointer)
        .def("print", &cppTensorDouble::print)
        .def("test", &cppTensorDouble::test);
    
    m.def("test_matMul_gpu", &cppTensor_Functions::test_matMul_gpu);
    m.def("add_cpu", &cppTensor_Functions::add_cpu<double>);
    m.def("add_gpu", &cppTensor_Functions::add_gpu<double>);
    m.def("transpose_cpu", &cppTensor_Functions::transpose_cpu<double>);    
    m.def("transpose_gpu", &cppTensor_Functions::transpose_gpu<double>);
    m.def("matMul_cpu", &cppTensor_Functions::matMul_cpu<double>);
    m.def("matMul_gpu", &cppTensor_Functions::matMul_gpu<double>);
    m.def("transpose_matMul", &cppTensor_Functions::transpose_matMul<double>);
    m.def("transpose_matMul_gpu", &cppTensor_Functions::transpose_matMul_gpu<double>);

    using CPPAdagradDouble = cppAdagrad<double>;
    pb11::class_<CPPAdagradDouble>(m, "cppAdagrad")
        .def(pb11::init<std::unordered_map<std::string, cppTensor<double>*>, double>())
        .def("zero_grad", &CPPAdagradDouble::zero_grad)
        .def("step", &CPPAdagradDouble::step)
        .def("test", &CPPAdagradDouble::test);

    using CPPCrossEntropyLossDouble = cppCrossEntropyLoss<double>;
    pb11::class_<CPPCrossEntropyLossDouble>(m, "cppCrossEntropyLoss")
        .def(pb11::init<>())
        .def("__call__", [](CPPCrossEntropyLossDouble& self, const cppTensor<double>& outputs, const cppTensor<double>& labels)
            {
                return self(outputs, labels);
            })
        .def("backward", &CPPCrossEntropyLossDouble::backward);

    using CPPRNNDouble = cppRnn<double>;
    pb11::class_<CPPRNNDouble>(m, "cppRnn")
        .def(pb11::init<double, const CPPRNNDouble::cppTensorType&, const CPPRNNDouble::cppTensorType&, const CPPRNNDouble::cppTensorType&, const CPPRNNDouble::cppTensorType&, 
            int, int, int, int>())
        .def("cuda", &CPPRNNDouble::cuda)
        .def("cpu", &CPPRNNDouble::cpu)
        .def("useSharedMemory", &CPPRNNDouble::useSharedMemory)
        .def("notUseSharedMemory", &CPPRNNDouble::notUseSharedMemory)
        .def("forward", &CPPRNNDouble::forward)
        .def("parameters", &CPPRNNDouble::parameters)
        .def("backward", &CPPRNNDouble::backward)
        .def("test", &CPPRNNDouble::test);

    using CPPLSTMDouble = cppLSTM<double>;
    pb11::class_<CPPLSTMDouble>(m, "cppLSTM")
        .def(pb11::init<>())
        .def("cuda", &CPPLSTMDouble::cuda)
        .def("cpu", &CPPLSTMDouble::cpu)
        .def("useSharedMemory", &CPPLSTMDouble::useSharedMemory)
        .def("notUseSharedMemory", &CPPLSTMDouble::notUseSharedMemory)
        .def("forward", &CPPLSTMDouble::forward)
        .def("backward", &CPPLSTMDouble::backward);

    using CPPGRUDouble = cppGRU<double>;
    pb11::class_<CPPGRUDouble>(m, "cppGRU")
        .def(pb11::init<>())
        .def("cuda", &CPPGRUDouble::cuda)
        .def("cpu", &CPPGRUDouble::cpu)
        .def("useSharedMemory", &CPPGRUDouble::useSharedMemory)
        .def("notUseSharedMemory", &CPPGRUDouble::notUseSharedMemory)
        .def("forward", &CPPGRUDouble::forward)
        .def("backward", &CPPGRUDouble::backward);
}
#include <torch/extension.h>
#include <cppTensor.hpp>
#include <cppRnn.hpp>

namespace pb11 = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using cppTensorDouble = cppTensor<double>;
    pb11::class_<cppTensorDouble>(m, "cppTensor")
        .def(pb11::init<>())
        .def(pb11::init<size_t, size_t>())
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

    using CPPRNNDouble = cppRnn<double>;
    pb11::class_<CPPRNNDouble>(m, "cppRnn")
        .def(pb11::init<double, const CPPRNNDouble::cppTensorType&, const CPPRNNDouble::cppTensorType&, const CPPRNNDouble::cppTensorType&, const CPPRNNDouble::cppTensorType&, 
            int, int, int, int>())
        .def("cuda", &CPPRNNDouble::cuda)
        .def("cpu", &CPPRNNDouble::cpu)
        .def("useSharedMemory", &CPPRNNDouble::useSharedMemory)
        .def("notUseSharedMemory", &CPPRNNDouble::notUseSharedMemory)
        .def("forward", &CPPRNNDouble::forward)
        .def("cross_entropy_loss", &CPPRNNDouble::cross_entropy_loss)
        .def("backward", &CPPRNNDouble::backward)
        .def("optimizer", &CPPRNNDouble::optimizer)
        .def("test", &CPPRNNDouble::test);
}
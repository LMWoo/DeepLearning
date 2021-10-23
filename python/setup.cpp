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
    
    m.def("test_dot_gpu", &cppTensor_Functions::test_dot_gpu);
    m.def("transpose_cpu", &cppTensor_Functions::transpose_cpu<double>);    
    m.def("dot_cpu", &cppTensor_Functions::dot_cpu<double>);
    m.def("dot_gpu", &cppTensor_Functions::dot_gpu<double>);

    using CPPRNNDouble = cppRnn<double>;
    pb11::class_<CPPRNNDouble>(m, "cppRnn")
        .def(pb11::init<double, const CPPRNNDouble::cppTensorType&, const CPPRNNDouble::cppTensorType&, const CPPRNNDouble::cppTensorType&, const CPPRNNDouble::cppTensorType&, 
            int, int, int, int>())
        .def("cuda", &CPPRNNDouble::cuda)
        .def("cpu", &CPPRNNDouble::cpu)
        .def("forward", &CPPRNNDouble::forward)
        .def("cross_entropy_loss", &CPPRNNDouble::cross_entropy_loss)
        .def("test", &CPPRNNDouble::test);
}
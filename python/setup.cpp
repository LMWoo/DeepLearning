#include <torch/extension.h>
#include <NumTest.hpp>
#include <cppRnn.hpp>

namespace pb11 = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using numTestDouble = numTest<double>;
    pb11::class_<numTestDouble>(m, "numTest")
        .def(pb11::init<>())
        .def(pb11::init<size_t, size_t>())
        .def(pb11::init<numTestDouble::numpyArray>())
        .def("cuda", &numTestDouble::cuda)
        .def("cpu", &numTestDouble::cpu)
        .def("numpy", &numTestDouble::numpy)
        .def("zeros", &numTestDouble::zeros)
        .def("ones", &numTestDouble::ones)
        .def("print_pointer", &numTestDouble::print_pointer)
        .def("print", &numTestDouble::print)
        .def("test", &numTestDouble::test);
    
    m.def("test_dot_gpu", &numTest_Functions::test_dot_gpu);
    m.def("transpose_cpu", &numTest_Functions::transpose_cpu<double>);    
    m.def("dot_cpu", &numTest_Functions::dot_cpu<double>);
    m.def("dot_gpu", &numTest_Functions::dot_gpu<double>);

    using CPPRNNDouble = cppRnn<double>;
    pb11::class_<CPPRNNDouble>(m, "cppRnn")
        .def(pb11::init<double, const CPPRNNDouble::numTestType&, const CPPRNNDouble::numTestType&, const CPPRNNDouble::numTestType&, const CPPRNNDouble::numTestType&, 
            int, int, int, int>())
        .def("cuda", &CPPRNNDouble::cuda)
        .def("cpu", &CPPRNNDouble::cpu)
        .def("forward", &CPPRNNDouble::forward)
        .def("cross_entropy_loss", &CPPRNNDouble::cross_entropy_loss)
        .def("test", &CPPRNNDouble::test);
}
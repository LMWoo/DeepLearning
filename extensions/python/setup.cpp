#include <torch/extension.h>
#include <NumCpp.hpp>

using namespace nc;
using namespace nc::pybindInterface;
namespace pb11 = pybind11;


torch::Tensor sigmoid_add(torch::Tensor x, torch::Tensor y)
{
    return x.sigmoid() + y.sigmoid();
}

namespace NdArrayInterface
{
    template<typename dtype>
    pbArrayGeneric ones(NdArray<dtype>& self)
    {
        self.ones();
        return nc2pybind<dtype>(self);
    }
}

namespace FunctionsInterface
{
    template<typename dtype>
    pbArrayGeneric onesRowCol(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2pybind(ones<dtype>(inNumRows, inNumCols));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sigmoid_add", &sigmoid_add, "sigmoid(x) + sigmoid(y)");

    pb11::class_<Shape>(m, "Shape")
        // .def(pb11::init<>())
        // .def(pb11::init<uint32>())
        .def(pb11::init<uint32, uint32>())
        .def(pb11::init<Shape>());
    
    using NdArrayDouble = NdArray<double>;

    pb11::class_<NdArrayDouble>(m, "NdArray")
        .def(pb11::init<Shape>())
        .def("ones", &NdArrayInterface::ones<double>)
        .def("print", &NdArrayDouble::print);


    // py::class_<NdArrayDouble>(m, "NdArray")
    //     .def(py::init<>())
    //     .def(py::init<Shape>())
    //     .def("ones", &NdArrayInterface::ones<double>)
    // m.def("onesRowCol", &FunctionsInterface::onesRowCol<double>, "onesRowCol");
}
#include <torch/extension.h>
#include <NumCpp.hpp>
#include <NumCpp/PythonInterface/PybindInterface.hpp>

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

    template<typename dtype>
    pbArrayGeneric getNumpyArray(NdArray<dtype>& inArray)
    {
        return nc2pybind<dtype>(inArray);
    }
}

namespace FunctionsInterface
{
    template<typename dtype>
    pbArrayGeneric onesRowCol(uint32 inNumRows, uint32 inNumCols)
    {
        return nc2pybind(ones<dtype>(inNumRows, inNumCols));
    }

    template<typename dtype1, typename dtype2>
    pbArrayGeneric dot(const NdArray<dtype1>& inArray1, const NdArray<dtype2>& inArray2)
    {
        return nc2pybind(nc::dot(inArray1, inArray2));
    }

    template<typename dtype>
    void memoryFree(NdArray<dtype> inArray)
    {
        inArray.memoryFree();
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
        .def(pb11::init<>())
        .def(pb11::init<Shape>())
        .def(pb11::init<pbArray<double>&>())
        .def("ones", &NdArrayInterface::ones<double>)
        .def("print", &NdArrayDouble::print)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<double>);


    m.def("dot", &nc::dot<double>);
    m.def("zeros_like", &zeros_like<double, double>);
    m.def("toNumCpp", &pybind2nc<double>);
    m.def("memoryFree", &FunctionsInterface::memoryFree<double>);
    // py::class_<NdArrayDouble>(m, "NdArray")
    //     .def(py::init<>())
    //     .def(py::init<Shape>())
    //     .def("ones", &NdArrayInterface::ones<double>)
    // m.def("onesRowCol", &FunctionsInterface::onesRowCol<double>, "onesRowCol");
}
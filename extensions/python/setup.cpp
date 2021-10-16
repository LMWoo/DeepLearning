#include <torch/extension.h>
#include <NumCpp.hpp>
#include <NumCpp/PythonInterface/PybindInterface.hpp>
#include <test_gpu.h>
#include <nnCpp.hpp>

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

namespace NNCppInterface
{
    
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

template<typename dtype>
class bindTest
{
private:
    int count{0};
    dtype test{0.0};
    dtype* p{nullptr};

    bool autoMemoryFree{true};
public:
    bindTest()
    {
        p = (dtype*)malloc(sizeof(dtype));
        printf("%d\n", p);
        printf("bindTest()\n");
    }
    bindTest(bool autoMemoryFree) 
    {
        this->autoMemoryFree = autoMemoryFree;
        p = (dtype*)malloc(sizeof(dtype));
        printf("%d\n", p);
        printf("bindTest()\n");
    }

    void memoryFree()
    {
        if (!autoMemoryFree && p)
        {
            free(p);
            printf("%d\n", p);
            printf("memoryFree()\n");
            p = nullptr;
        }
    }
    ~bindTest()
    {
        if (autoMemoryFree && p)
        {
            free(p);
            printf("%d\n", p);
            printf("~bindTest()\n");
            p = nullptr;
        }
    }

    void print()
    {
        printf("%lf\n", test);
    }

    bindTest<dtype> return_test()
    {
        return bindTest<dtype>(false);
    }
};

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
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<double>)
        .def("memoryFree", &NdArrayDouble::memoryFree)
        .def("dot", &NdArrayDouble::dot);

    m.def("dot", &nc::dot<double>);
    m.def("zeros_like", &zeros_like<double, double>);
    m.def("toNumCpp", &pybind2nc<double>);
    m.def("memoryFree", &memory::memoryFree);
    
    m.def("test_gpu", &test_gpu::test);
    m.def("test_gpu_matrix_add", &test_gpu::test_matrix_add);

    using RNNDouble = nnCpp::rnn<double>;
    
    pb11::class_<RNNDouble>(m, "rnn")
        .def(pb11::init<double, uint32, uint32, uint32, uint32>())
        .def("test", &RNNDouble::test)
        .def("forward", &RNNDouble::forward)
        .def("cross_entropy_loss", &RNNDouble::cross_entropy_loss)
        .def("backward", &RNNDouble::backward)
        .def("softmax", &RNNDouble::softmax)
        .def("deriv_softmax", &RNNDouble::deriv_softmax)
        .def("optimizer", &RNNDouble::optimizer);
    
    pb11::class_<bindTest<double>>(m, "bindTest")
        .def(pb11::init<>())
        .def("print", &bindTest<double>::print)
        .def("return_test", &bindTest<double>::return_test)
        .def("memoryFree", &bindTest<double>::memoryFree);
}
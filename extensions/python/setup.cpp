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
    dtype* p{nullptr};
    dtype* back_p{nullptr};
    size_t size_{0};

public:
    bindTest(size_t size) :
        size_(size)
    {
        initialize();
        
    }

    ~bindTest()
    {
        if (p)
        {
            free(p);
            printf("print start ~bindTest\n");
            printf("this : %p\n", this);
            printf("p : %d\n", p);
            printf("print end ~bindTest()\n");
            p = nullptr;
        }
    }

    void initialize()
    {
        if (!p)
        {
            p = (dtype*)malloc(sizeof(dtype) * size_);
        }
        printf("print start initialize\n");
        printf("this : %p\n", this);
        printf("p : %d\n", p);

        for (size_t i = 0; i < size_; ++i)
        {
            p[i] = size_ * 10;
        }

        this->print();

        printf("print end initialize\n");
    }

    bindTest<dtype> clone()
    {
        back_p = p;
        p = nullptr;
        return *this;
    }

    void print()
    {
        for (size_t i = 0; i < size_; ++i)
        {
            std::cout << p[i] << " ";
        }
        std::cout << "\n";
    }

    void useArray()
    {
        p = back_p;
        back_p = nullptr;
    }

    bindTest<dtype> forward()
    {
        bindTest<dtype> returnArray(20);
        return return_test(returnArray.clone());
    }

    bindTest<dtype> return_test(bindTest<dtype> input_)
    {
        input_.useArray();

        printf("print start return_test\n");
        // printf("this : %p\n", input_);
        printf("p : %d\n", input_.p);

        input_.print();

        printf("print end return_test\n");

        return input_.clone();
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
        .def("initialize", &NdArrayDouble::initialize)
        .def("ones", &NdArrayInterface::ones<double>)
        .def("print", &NdArrayDouble::print)
        .def("getNumpyArray", &NdArrayInterface::getNumpyArray<double>)
        .def("dot", &NdArrayDouble::dot);

    m.def("dot", &nc::dot<double>);
    m.def("zeros_like", &zeros_like<double, double>);
    m.def("toNumCpp", &pybind2nc<double>);
    
    m.def("test_gpu", &test_gpu::test);
    m.def("test_gpu_matrix_add", &test_gpu::test_matrix_add);

    m.def("memoryClean", &memory::memoryClean);

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
        .def(pb11::init<size_t>())
        .def("initialize", &bindTest<double>::initialize)
        .def("useArray", &bindTest<double>::useArray)
        .def("forward", &bindTest<double>::forward)
        .def("print", &bindTest<double>::print)
        .def("return_test", &bindTest<double>::return_test);
}
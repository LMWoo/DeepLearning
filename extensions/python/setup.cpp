#include <torch/extension.h>
#include <NumCpp.hpp>

using namespace nc;


// test include_dirs in setuptools.setup with relative path

torch::Tensor sigmoid_add(torch::Tensor x, torch::Tensor y) {
  return x.sigmoid() + y.sigmoid();
}

struct MatrixMultiplier {
  MatrixMultiplier(int A, int B) {
    tensor_ =
        torch::ones({A, B}, torch::dtype(torch::kFloat64).requires_grad(true));
  }
  torch::Tensor forward(torch::Tensor weights) {
    return tensor_.mm(weights);
  }
  torch::Tensor get() const {
    return tensor_;
  }

 private:
  torch::Tensor tensor_;
};

bool function_taking_optional(c10::optional<torch::Tensor> tensor) {
  return tensor.has_value();
}

namespace NdArrayInterface
{
  // template<typename dtype>
  // pbArrayGeneric partition(NdArray<dtype>& self, uint32 inKth, Axis inAxis = Axis::None)
  // {
  //   self.partition(inKth, inAxis);
  //   return nc2pybind<dtype>(self);
  // }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid_add", &sigmoid_add, "sigmoid(x) + sigmoid(y)");
  m.def(
      "function_taking_optional",
      &function_taking_optional,
      "function_taking_optional");
  py::class_<MatrixMultiplier>(m, "MatrixMultiplier")
      .def(py::init<int, int>())
      .def("forward", &MatrixMultiplier::forward)
      .def("get", &MatrixMultiplier::get);

  // py::class_<NdArray>(m, "NdArray")
  //  .def(py::init<>());
    //.def("ones", &NdArrayInterface::partition<double>);
}
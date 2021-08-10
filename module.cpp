#include <Python.h>
#include <vector>
#include <iostream>

#define C10_EXPORT __attribute__((__visibility__("default")))
#define C10_IMPORT C10_EXPORT
#define TORCH_API C10_IMPORT

static std::vector<PyMethodDef> methods;

PyObject* module;

static PyObject * THPModule_initExtension(PyObject *_unused, PyObject *shm_manager_path)
{
    std::cout << "Hell" << std::endl;
    Py_RETURN_NONE;
}

static PyMethodDef TorchMethods[] = {
    {"_initExtension", THPModule_initExtension, METH_O, nullptr},
};

void THPUtils_addPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods)
{
    if (!vector.empty())
    {
        vector.pop_back();
    }
    while(true)
    {
        vector.push_back(*methods);
        if (!methods->ml_name) {
            break;
        }
        methods++;
    }
}

TORCH_API PyObject* initModule();

PyObject* initModule()
{
#define ASSERT_TRUE(cmd) if (!(cmd)) return nullptr

    THPUtils_addPyMethodDefs(methods, TorchMethods);
    static struct PyModuleDef torchmodule = {
        PyModuleDef_HEAD_INIT,
        "torch._C",
        nullptr,
        -1,
        methods.data()
    };
    ASSERT_TRUE(module = PyModule_Create(&torchmodule));
    return module;
}
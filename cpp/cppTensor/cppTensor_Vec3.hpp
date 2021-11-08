#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <cuda.h>
#include "cppTensor_gpu.hpp"

template<typename dtype>
class cppTensor_Vec3
{
public:
    using numpyArray = pybind11::array_t<dtype, pybind11::array::c_style>;
    using numpyArrayGeneric = pybind11::array;

public:
    class Vec3
    {
    public:
        int z{0};
        int y{0};
        int x{0};

        int size() const
        {
            return z * y * x;
        }

        void print()
        {
            printf("shape z %d y %d x %d\n", z, y, x);
        }

        Vec3() = default;

        Vec3(int z, int y, int x)
        {
            this->z = z;
            this->y = y;
            this->x = x;
        }

        Vec3(const Vec3& rhs)
        {
            this->z = rhs.z;
            this->y = rhs.y;
            this->x = rhs.x;
        }

        int& operator[](int idx)
        {
            if (idx == 0)
            {
                return z;
            }
            else if (idx == 1)
            {
                return y;
            }
            else if (idx == 2)
            {
                return x;
            }
        }
    };
public:
    dtype* data_{nullptr};
    dtype* dev_data_{nullptr};
    Vec3 shape_;
    bool is_cuda_{false};
    bool is_owner_{true};

private:
    void newArray()
    {
        deleteArray();
        data_ = (dtype*)malloc(sizeof(dtype) * shape_.size());
        this->zeros();

        print_pointer("newArray()");
    }

    void newArray(int z, int y, int x)
    {
        shape_ = Vec3(z, y, x);
        newArray();
    }

    void deleteArray()
    {
        print_pointer("deleteArray()");

        if (this->is_owner_)
        {
            cppTensor_gpu::cpu_free(data_);
            data_=nullptr;
            cppTensor_gpu::gpu_free(dev_data_);
            dev_data_=nullptr;
        }
    }

public:
    cppTensor_Vec3() = default;

    cppTensor_Vec3(int z, int y, int x) :
        shape_(Vec3(z, y, x))
    {
        newArray();
    }

    cppTensor_Vec3(int z, int y, int x, bool is_cuda) :
        shape_(Vec3(z, y, x))
    {
        if (is_cuda)
        {
            this->shape_ = Vec3(z, y, x);
            this->is_cuda_ = is_cuda;
            this->dev_data_ = cppTensor_gpu::gpu_malloc(shape_.size() * sizeof(double));
        }
    }

    cppTensor_Vec3(const cppTensor_Vec3<dtype>& rhs)
    {
        this->shape_ = Vec3(rhs.shape_);
        newArray();

        if (rhs.is_cuda_)
        {
            this->cuda();
            cppTensor_gpu::copy_gpu_to_gpu(this->shape_.size() * sizeof(double), this->dev_data_, rhs.dev_data_);
        }
        else
        {
            std::copy(rhs.data_, rhs.data_ + rhs.shape_.size(), this->data_);
        }
    }
    
    cppTensor_Vec3<dtype>& operator=(const cppTensor_Vec3<dtype>& rhs)
    {
        this->shape_ = Vec3(rhs.shape_);
        newArray();

        if (rhs.is_cuda_)
        {
            this->cuda();
            cppTensor_gpu::copy_gpu_to_gpu(this->shape_.size() * sizeof(double), this->dev_data_, rhs.dev_data_);
        }
        else
        {
            std::copy(rhs.data_, rhs.data_ + rhs.shape_.size(), this->data_);
        }

        return *this;
    }


    cppTensor_Vec3(cppTensor_Vec3<dtype>&& rhs) noexcept
    {
        deleteArray();
        
        this->data_ = rhs.data_;
        this->dev_data_ = rhs.dev_data_;
        this->shape_ = Vec3(rhs.shape_);
        this->is_cuda_= rhs.is_cuda_;

        rhs.data_ = nullptr;
        rhs.dev_data_ = nullptr;
    }

    cppTensor_Vec3<dtype>& operator=(cppTensor_Vec3<dtype>&& rhs) noexcept
    {
        deleteArray();

        this->data_ = rhs.data_;
        this->dev_data_ = rhs.dev_data_;
        this->shape_ = Vec3(rhs.shape_);
        this->is_cuda_= rhs.is_cuda_;

        rhs.data_ = nullptr;
        rhs.dev_data_ = nullptr;

        return *this;
    }

    cppTensor_Vec3(const numpyArray& numpyInput)
    {
        const auto dataPtr = numpyInput.data();

        switch(numpyInput.ndim())
        {
            case 0:
            {
                break;
            }
            case 1:
            {
                const int size = static_cast<int>(numpyInput.size());
                shape_ = Vec3(1, 1, size);
                newArray();
                std::copy(dataPtr, dataPtr + shape_.size(), this->data_);
                break;
            }
            case 2:
            {
                const int rows = static_cast<int>(numpyInput.shape(0));
                const int cols = static_cast<int>(numpyInput.shape(1));
                shape_ = Vec3(1, rows, cols);
                newArray();
                std::copy(dataPtr, dataPtr + shape_.size(), this->data_);
            }
            case 3:
            {
                const int z = static_cast<int>(numpyInput.shape(0));
                const int y = static_cast<int>(numpyInput.shape(1));
                const int x = static_cast<int>(numpyInput.shape(2));
                shape_ = Vec3(z, y, x);
                newArray();
                std::copy(dataPtr, dataPtr + shape_.size(), this->data_);
            }
            default:
            {
                break;
            }
        }
    }

    ~cppTensor_Vec3()
    {
        deleteArray();
    }

public:
    cppTensor_Vec3<dtype>& fill(dtype value)
    {
        std::fill(this->data_, this->data_ + this->shape_.size(), value);
        return *this;
    }

    dtype& operator()(int z, int y, int x) noexcept
    {
        return data_[shape_.x * shape_.y * z + shape_.x * y + x];
    }

    const dtype& operator()(int z, int y, int x) const noexcept
    {
        return data_[shape_.x * shape_.y * z + shape_.x * y + x];
    }

    void zeros()
    {
        if (is_cuda_)
        {
            cppTensor_gpu::zeros_gpu(this->dev_data_, sizeof(double) * this->shape_.size());
        }
        else
        {
            this->fill(dtype{0});
        }
    }

    void ones()
    {
        this->fill(dtype{1});
    }

    numpyArrayGeneric numpy()
    {
        const std::vector<pybind11::ssize_t> numpy_shape{
            static_cast<pybind11::ssize_t>(shape_.z),
            static_cast<pybind11::ssize_t>(shape_.y), 
            static_cast<pybind11::ssize_t>(shape_.x)};
        const std::vector<pybind11::ssize_t> numpy_strides{
            static_cast<pybind11::ssize_t>(shape_.x * shape_.y * sizeof(dtype)),
            static_cast<pybind11::ssize_t>(shape_.x * sizeof(dtype)),
            static_cast<pybind11::ssize_t>(sizeof(dtype))};
        return numpyArrayGeneric(numpy_shape, numpy_strides, this->data_);
    }

    std::string str()
    {
        std::string out;
        for (int zz = 0; zz < shape_.z; ++zz)
        {
            for (int yy = 0; yy < shape_.y; ++yy)
            {
                for (int xx = 0; xx < shape_.x; ++xx)
                {
                    out += std::to_string(operator()(zz, yy, xx)) + ", ";
                }
                out += "\n";
            }
            out += "\n";
        }
        return out;
    }

    void print_pointer(std::string str)
    {
        PRINT_DEBUG("call by %s this %p data %p\n", str.c_str(), this, data_);
        PRINT_DEBUG("call by %s this %p dev_data %p\n", str.c_str(), this, dev_data_);
    }

    void print()
    {
        if (is_cuda_)
        {
            this->cpu();
            std::cout << str();
            this->cuda();
        }
        else
        {
            std::cout << str();
        }
    }

    void cuda()
    {
        if (dev_data_ == nullptr) 
        {
            dev_data_ = cppTensor_gpu::gpu_malloc(shape_.size() * sizeof(double));
            cppTensor_gpu::copy_cpu_to_gpu(shape_.size() * sizeof(double), dev_data_, data_);
            cppTensor_gpu::cpu_free(data_);

            print_pointer("cuda()");
            
            data_=nullptr;
            is_cuda_ = true;
        }
    }

    void cpu()
    {
        if (dev_data_)
        {
            data_ = (dtype*)cppTensor_gpu::cpu_malloc(shape_.size() * sizeof(dtype));
            cppTensor_gpu::copy_gpu_to_cpu(shape_.size() * sizeof(double), data_, dev_data_);
            cppTensor_gpu::gpu_free(dev_data_);

            print_pointer("cpu()");

            dev_data_ = nullptr;
            is_cuda_ = false;
        }
    }

    void test()
    {
        printf("cppTensor_Vec3 test\n");
    }
};
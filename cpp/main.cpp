// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <iostream>
#include <iomanip>

#include <NumCpp.hpp>
#include <memory>

#include "test_gpu.h"
#include <chrono>
#include "nnCpp.hpp"

void torch_cpp_example();
void gpu_example();

int main() {
    
    // gpu_example();

    nc::NdArray<double> lhs = nc::random::rand(0.0, 1.0, nc::Shape(3, 4));
    nc::NdArray<double> rhs = nc::random::rand(1.0, 2.0, nc::Shape(3, 4));
    
    auto result = nc::dot(lhs, rhs);

    // printf("lhs\n");
    // lhs.print();
    // printf("rhs\n");
    // rhs.print();

    // printf("+\n");
    // (lhs + rhs).print();

    // printf("-\n");
    // (lhs - rhs).print();

    // printf("*\n");
    // (lhs * 3.0).print();

    nc::NdArray<double> sum_test = nc::random::rand(0.0, 1.0, nc::Shape(3, 1));
    sum_test.print();
    
    nc::NdArray<double> sum_result = nc::sum<double>(sum_test);
    printf("%lf\n", *sum_result.begin());


    nnCpp::rnn<double> model(0.01, 28, 28, 128, 10);
    model.softmax(sum_test).print();

    // nc::NdArray<double> hprev = nc::zeros<double>(128, 1);
    // nc::NdArray<double> images[28];

    // for (int i = 0; i < 28; ++i)
    // {
    //     images[i] = nc::zeros<double>(1, 28);
    // }

    // nc::NdArray<double> output = model.forward(images, hprev);   
    // nc::NdArray<double>** gradients = model.backward(output);
    // model.optimizer(gradients);

    // int a = 1;
    // int b = 2;

    // int* c[2] = {&a, &b};
    // *c[0] = 10;
    // printf("%d\n", a);

    // torch_cpp_example();
    // nc::ones<int>(3, 4);
    // nc::Shape shape(3, 3);
    // nc::NdArray<double> ndarray(shape);
    // ndarray = ndarray.ones();
    // ndarray.print();

    return 0;
}

void gpu_example()
{
    test_gpu::test();
    test_gpu::test_matrix_add();
    
    nc::NdArray<double> X(512, 512);
    nc::NdArray<double> Y(512, 512);

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    test_gpu::test_matrix_mul();
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::nanoseconds elapsedNS = end-start;
    std::chrono::seconds elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(elapsedNS);
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time NS: " << elapsedNS.count() << "ns\n"
              << "elapsed time S: " << elapsedSeconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    for (int i = 0; i < 128; ++i)
        X.dot(Y);
    end = std::chrono::system_clock::now();
    elapsedNS = end-start;
    elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(elapsedNS);
    end_time = std::chrono::system_clock::to_time_t(end);
    
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time NS: " << elapsedNS.count() << "ns\n"
              << "elapsed time S: " << elapsedSeconds.count() << "s\n";
}

void torch_cpp_example()
{
    std::cout << "Logistic Regression\n\n";

    // Device
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Hyper parameters
    const int64_t input_size = 784;
    const int64_t num_classes = 10;
    const int64_t batch_size = 100;
    const size_t num_epochs = 5;
    const double learning_rate = 0.001;

    const std::string MNIST_data_path = "/media/lee/ESD-ISO/study/data/mnist/";

    // MNIST Dataset (images and labels)
    auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();

    auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();

    // Data loaders
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

    // Logistic regression model
    torch::nn::Linear model(input_size, num_classes);

    model->to(device);

    // Loss and optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate));

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "Training...\n";

    // Train the model
    for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
        // Initialize running metrics
        double running_loss = 0.0;
        size_t num_correct = 0;

        for (auto& batch : *train_loader) {
            auto data = batch.data.view({batch_size, -1}).to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model->forward(data);

            // Calculate loss
            auto loss = torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss.item<double>() * data.size(0);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum().item<int64_t>();

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(num_correct) / num_train_samples;

        std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }

    std::cout << "Training finished!\n\n";
    std::cout << "Testing...\n";

    // Test the model
    model->eval();
    torch::NoGradGuard no_grad;

    double running_loss = 0.0;
    size_t num_correct = 0;

    for (const auto& batch : *test_loader) {
        auto data = batch.data.view({batch_size, -1}).to(device);
        auto target = batch.target.to(device);

        auto output = model->forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, target);

        running_loss += loss.item<double>() * data.size(0);

        auto prediction = output.argmax(1);

        num_correct += prediction.eq(target).sum().item<int64_t>();
    }

    std::cout << "Testing finished!\n";

    auto test_accuracy = static_cast<double>(num_correct) / num_test_samples;
    auto test_sample_mean_loss = running_loss / num_test_samples;

    std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
}
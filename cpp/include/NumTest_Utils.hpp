#pragma once

#include <iostream>
#include <string>
#include <chrono>

namespace NumTest_Utils
{
    static std::chrono::system_clock::time_point start;
    static std::chrono::system_clock::time_point end;

    void time_start()
    {
        start = std::chrono::system_clock::now();
    }

    void time_end()
    {
        end = std::chrono::system_clock::now();

        std::chrono::nanoseconds elapsedNS = end-start;
        std::chrono::seconds elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(elapsedNS);
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    
        std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time NS: " << elapsedNS.count() << "ns\n"
            << "elapsed time S: " << elapsedSeconds.count() << "s\n";
    }

    void exception_print(std::string function_name, std::string exception_str)
    {
        printf("exception call by %s\n", function_name.c_str());   
        printf("%s\n", exception_str.c_str());
    }

    void null_check(std::string function_name, std::string pointer_name, void* ptr)
    {
        if (ptr == nullptr)
        {
            exception_print(function_name, pointer_name + " == nullptr");
        }
    }
}
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <ratio>
#include "../../../include/similarity/cosine_sim.hpp"

float cosine_similarity_cpu(float* vecA, float* vecB, int n_dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < n_dim; i++) {
        dot += vecA[i] * vecB[i];
        norm_a += vecA[i] * vecA[i];
        norm_b += vecB[i] * vecB[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    return dot / (norm_a * norm_b);
}


int main(){

    float* vectA;
    float* vectB;
    int n_dim = 512;

    vectA = (float*)malloc(n_dim * sizeof(float));
    vectB = (float*)malloc(n_dim * sizeof(float));
    
    for(int i = 0 ; i < n_dim ; i++){
        vectA[i] = rand() % 100;
        vectB[i] = rand() % 100;
    }

    auto gpu_st = std::chrono::high_resolution_clock::now();
    
    float gpu_result = cosine_similarity(vectA, vectB, n_dim);
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_st);

    auto cpu_st = std::chrono::high_resolution_clock::now();

    float cpu_result = cosine_similarity_cpu(vectA, vectB, n_dim);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_st);
    std::cout << "GPU Result: " << gpu_result << std::endl;
    std::cout << "CPU Result: " << cpu_result << std::endl;
    assert(std::abs(gpu_result - cpu_result) < 1e-5 && "GPU vs CPU mismatch");
    std::cout << "Test passed!" << std::endl;

    std::cout << "TOTAL TIME" << std::endl;
    std::cout << "CPU TIME :  " << std::chrono::duration<float>(cpu_time).count()  << std::endl;
    std::cout << "CPU TIME :  " << std::chrono::duration<float>(gpu_time).count() << std::endl;
    std::cout << "SpeedUp : " << std::chrono::duration<float>(cpu_time).count() / std::chrono::duration<float>(gpu_time).count() << std::endl;

}
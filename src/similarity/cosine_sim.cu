#include <cstddef>
#include <cublas_v2.h>

#include "../../include/similarity/cosine_sim.hpp"

/*
Cosine Similarity -> 
vector A , vector B
nominator -> A * B
denominator -> norm(A) * norm(B)

*/


float cosine_similarity(float* h_vectorA , float* h_vectorB , int n_dim){
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_vectorA , *d_vectorB;

    size_t size = n_dim * sizeof(float);

    cudaMalloc(&d_vectorA , size);
    cudaMalloc(&d_vectorB , size);

    cudaMemcpy(d_vectorA , h_vectorA , size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectorB , h_vectorB , size , cudaMemcpyHostToDevice);
    
    float norm_a , norm_b;
    cublasSnrm2(handle , n_dim , d_vectorA ,1 , &norm_a);
    cublasSnrm2(handle , n_dim , d_vectorB ,1 , &norm_b);

    float dot_product_value;
    cublasSdot(handle , n_dim , d_vectorA , 1 , d_vectorB ,1 , &dot_product_value);

    float similarity = dot_product_value / (norm_a * norm_b);

    return similarity;

}
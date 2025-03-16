#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cmath>
#include <cstddef>
#include <cublas_v2.h>
#include <shared_mutex>

#include "../../include/similarity/cosine_sim.hpp"
#include "../../include/similarity/cosine_sim.cuh"


/*
Cosine Similarity -> 
vector A , vector B
nominator -> A * B
denominator -> norm(A) * norm(B)

*/


float cosine_similarity(float* d_vectorA , float* d_vectorB , int n_dim){
    cublasHandle_t handle;
    cublasCreate(&handle);

    float norm_a , norm_b;
    cublasSnrm2(handle , n_dim , d_vectorA ,1 , &norm_a);
    cublasSnrm2(handle , n_dim , d_vectorB ,1 , &norm_b);

    float dot_product_value;
    cublasSdot(handle , n_dim , d_vectorA , 1 , d_vectorB ,1 , &dot_product_value);

    float similarity = dot_product_value / (norm_a * norm_b);

    return similarity;

}


__global__
void cosine_similarity_kernel(
    const float *space_storage,
    const float *search_vector,
    float search_norm,
    float *similarity_val,
    int num_vectors,
    int dim
){

    //one block per stored vector
    int vec_idx = blockIdx.x;
    int tid = threadIdx.x;
    if(vec_idx >= num_vectors) return;

    cublasHandle_t handle;
    cublasCreate(&handle);
    float dot_product_value;
    cublasSdot(handle , dim , search_vector , 1 , &space_storage[vec_idx * dim] ,1 , &dot_product_value);

    float norm_vector;
    cublasSnrm2(handle , dim , &space_storage[vec_idx * dim] ,1 , &norm_vector);

    if(tid == 0){
        similarity_val[vec_idx] = dot_product_value / (norm_vector * search_norm + 1e-6f); //adding small num to avoid division by 0
    }   

}



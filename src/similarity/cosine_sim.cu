#include <cublas_v2.h>
#include "../../include/similarity/cosine_sim.hpp"

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
    float* space_storage,
    const float* search_vector,
    float *similarity_val,
    int num_vectors,
    int dim
){

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int vector_index = blockIdx.x;


    float dot_product = 0.0f;

    float search_norm_sq = 0.0f;
    float storage_norm_sq = 0.0f;

    if(tid < dim){
        sdata[tid] = search_vector[tid];
    }
    __syncthreads();

    for(int i = tid ; i < dim ; i += blockDim.x){
        float sv = sdata[i];
        float store_v = space_storage[vector_index * dim + i];
        
        dot_product += sv * store_v;
        search_norm_sq += sv * sv;
        storage_norm_sq += store_v * store_v;

        
    }


    __shared__ float temp_dot[256];
    temp_dot[tid] = dot_product;
    __syncthreads();

    for(int s = blockDim.x / 2 ; s > 0 ; s>>=1){
        if (tid < s){
            temp_dot[tid] += temp_dot[tid + s];
        }
        __syncthreads();
    }



    //reduce search norm;
    __shared__ float temp_search[256];
    temp_search[tid] = search_norm_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            temp_search[tid] += temp_search[tid + s];
        }
        __syncthreads();
    }

    //reduce storage norm
    __shared__ float temp_storage[256];
    temp_storage[tid] = storage_norm_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            temp_storage[tid] += temp_storage[tid + s];
        }
        __syncthreads();
    }



    if (tid == 0 && vector_index < num_vectors) {
        float denominator = sqrt(temp_search[0]) * sqrt(temp_storage[0]);
        if (denominator > 0) {
            similarity_val[vector_index] = temp_dot[0] / denominator;
        } else {
            similarity_val[vector_index] = 0.0f; 
        }
    }

}


void launch_cosine_similarity_kernel(
    float *space_storage,
    const float *search_vector,
    float search_norm,
    float *similarity_val,
    int num_vectors,
    int dim
){
    int blockSize = 256;
    int gridSize = num_vectors;
    size_t sharedMemSize = dim * sizeof(float);
    cosine_similarity_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        space_storage, 
        search_vector, 
        similarity_val, 
        num_vectors, 
        dim
    );
    cudaDeviceSynchronize();
}
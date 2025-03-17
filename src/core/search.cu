#include "../../include/core/search.hpp"
#include "../../include/similarity/cosine_sim.hpp"
#include <cstddef>
#include <algorithm>
#include <cstdlib>
#include <vector>

std::vector<float> _spaceFlat_search_cosine(
    eanns::Tensor* query_tensor , 
    std::vector<float>& vector_storage,
    int num_vectors,
    int dim,
    int n_top
){
    const float* query_vector = query_tensor->get_vector()->data();
    float* space_storage = vector_storage.data();

    //declare device variables
    float *d_space_storage , *similarity_val;
    const float * d_search_vector;
    size_t space_size = sizeof(float) * dim * num_vectors;
    size_t search_size = sizeof(float) * dim;
    size_t similarity_vector_size = sizeof(float) * num_vectors;
    
    cudaMalloc(&d_space_storage , space_size);
    cudaMalloc(&d_search_vector , search_size);
    cudaMalloc(&similarity_val , similarity_vector_size);

    cudaMemcpy(d_space_storage , space_storage , space_size , cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_search_vector , query_vector , search_size , cudaMemcpyHostToDevice);

    launch_cosine_similarity_kernel(
        d_space_storage,
        d_search_vector,
        similarity_val,
        num_vectors,
        dim
    );

    float* h_similarity_vals;

    h_similarity_vals = (float*)malloc(similarity_vector_size);
    cudaMemcpy(h_similarity_vals , similarity_val , similarity_vector_size , cudaMemcpyDeviceToHost);

    // auto maxIt = std::max_element(h_similarity_vals , h_similarity_vals + similarity_vector_size);
    // float maxValue = *maxIt;
    // int maxIndex = std::distance(h_similarity_vals , maxIt);

    std::sort(h_similarity_vals , h_similarity_vals + num_vectors);

    // float* return_sim_val = (float*)malloc(sizeof(float) * n_top);
    std::vector<float> return_sim_val;

    for(int i = 0 ; i < n_top ; i++){
        return_sim_val.push_back(h_similarity_vals[i]);
    }


    free(h_similarity_vals);
    cudaFree(d_space_storage);
    cudaFree(similarity_val);

    return return_sim_val;

}
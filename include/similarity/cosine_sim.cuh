#ifndef COSINE_SIM_CUH 
#define COSINE_SIM_CUH


__global__
void cosine_similarity_kernel(
    const float *space_storage,
    const float *search_vector,
    float search_norm,
    float *similarity_val,
    int num_vectors,
    int dim
);




#endif
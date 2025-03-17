#pragma once

void launch_cosine_similarity_kernel(
    float *space_storage,
    const float *search_vector,
    float *similarity_val,
    int num_vectors,
    int dim
);


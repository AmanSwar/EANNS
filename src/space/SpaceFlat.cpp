#include "../../include/space/SpaceFlat.hpp"
#include "../../include/similarity/cosine_sim.hpp"


SpaceFlat::SpaceFlat(
    uint64_t space_id,
    std::string space_name,
    std::string similarity_metric,
    std::string storage_type
    
){
    this->space_id = space_id;
    this->space_name = space_name;
    this->similarity_metric = similarity_metric;
    this->storage_type = storage_type;
}

void SpaceFlat::insert(eanns::Tensor& vector){
    storage.push_back(vector);
    vector_count++;

}


void SpaceFlat::search(eanns::Tensor& vector){
    /*
    Brute Force Search
    */

    float highest_similarity = 0;
    for(int index = 0 ; index < vector_count ; index++){


    }
}
    
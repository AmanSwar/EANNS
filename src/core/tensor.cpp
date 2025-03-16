#include "../../include/core/tensor.hpp"
#include <vector>


const std::vector<float>* eanns::Tensor::get_vector(){
    const std::vector<float>* ptr = &vector;
    return ptr;
}

const std::unordered_map<std::string , std::string>* eanns::Tensor::get_metdata(){
    const std::unordered_map<std::string , std::string>* ptr = &metadata;
    return ptr;
}


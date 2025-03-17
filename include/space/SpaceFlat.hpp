#pragma once
#include "space.hpp"
#include "../core/tensor.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class SpaceFlat: public Space
{
private:
    std::vector<eanns::Tensor> storage;
    std::vector<float> vector_storage;
    std::vector<std::unordered_map<std::string, std::string>> metadata_storage;
    int vector_count;
public:
    SpaceFlat(
        uint64_t space_id,
        std::string space_name,
        std::string similarity_metric,
        std::string storage_type
    );
    void insert(eanns::Tensor& vector) override;
    void search(eanns::Tensor& vector) override;

};



#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include "../core/tensor.hpp"

class Space{
protected:
    uint64_t space_id;
    std::string space_name;
    std::string similarity_metric;
    std::string storage_type;
    std::unordered_map<int, int> Space_table;

public:
    
    ~Space() = default;
    virtual void insert(eanns::Tensor& vector);
    virtual void search(eanns::Tensor& vector);
    virtual void remove(uint64_t id);


};


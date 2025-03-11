#pragma once
#include <unordered_map>
#include <string>

namespace eanns{

    struct Tensor{
        float *vector;
        std::unordered_map<std::string , std::string>* metadata;
    };

}

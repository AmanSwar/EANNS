#pragma once
#include <cstddef>
#include <unordered_map>
#include <string>
#include <vector>


namespace eanns{

    class Tensor{
        private:
            std::vector<float> vector;
            std::unordered_map<std::string , std::string> metadata;
        public:
            Tensor(
                std::vector<float> vector , 
                std::unordered_map<std::string , std::string> metadata = {}
            ) : vector(vector) , metadata(metadata) {};          
        };
}

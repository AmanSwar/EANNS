#pragma once
#include <unordered_map>
#include <string>
#include <vector>


namespace eanns{

    class Tensor{
        private:
            const std::vector<float> vector;
            const std::unordered_map<std::string , std::string> metadata;
        public:
            Tensor(
                std::vector<float> vector , 
                std::unordered_map<std::string , std::string> metadata = {}
            ) : vector(vector) , metadata(metadata) {};
            
            const std::vector<float>* get_vector();
            const std::unordered_map<std::string , std::string>* get_metdata();
        };
}

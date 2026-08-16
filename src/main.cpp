#include <iostream>
#include "../include/xorAlg.h"
#include <random>

int main() {

    std::random_device rnd_device;
    std::mt19937 engine {rnd_device()};  

    Neuron n(3, engine);
    for (auto w: n.weights)
        {std::cout << "weights: " << w << std::endl;}
    std::cout << "bias: " << n.bias << std::endl; 
    double outputs = n.activate({0,0.1,2});
    std::cout << outputs << std::endl;
    for (auto w: n.weights)
        {std::cout << "weights: " << w << std::endl;}
         std::cout << "bias: " << n.bias << std::endl; 
    return 0;
}
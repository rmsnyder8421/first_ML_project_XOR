#include <iostream>
#include "../include/xorAlg.h"
#include <random>

int main() {

    std::random_device rnd_device;
    std::mt19937 engine {rnd_device()};  

    double learning_rate = 0.1;
    size_t epochs = 100000;
    std::vector<std::vector<double>> inputs  = {{0,0}, {0,1}, {1,0}, {1,1}};
    std::vector<std::vector<double>> outputs = {{0},   {1},   {1},   {0}};

    //layers have args num_neurons, num_inputs, num_outputs, engine
    Layer hiddenLayer(2, 2, 1, engine);
    Layer outputLayer(1, 2, 1, engine);

    std::vector<Layer> layers = {hiddenLayer, outputLayer};
    
    NeuralNetwork nn(layers, learning_rate, epochs);
    nn.train(inputs, outputs);

    for (std::size_t l = 0; l < nn.layers.size(); l++)
    {
        std::cout << "\nLayer " << l << "\n";
        Layer& layer = nn.layers.at(l);

        for (std::size_t n = 0; n < layer.neurons.size(); n++)
        {
            Neuron& neuron = layer.neurons.at(n);
            std::cout << "  neuron " << n << "  w = [";
            for (std::size_t w = 0; w < neuron.weights.size(); w++)
                std::cout << (w ? ", " : "") << neuron.weights.at(w);
            std::cout << "]  b = " << neuron.bias << "\n";
        }
    }

    std::cout << "\nPredictions\n";
    for (std::size_t i = 0; i < inputs.size(); i++)
    {
        std::vector<double> out = nn.predict(inputs.at(i));
        std::cout << "  " << inputs.at(i).at(0) << " xor " << inputs.at(i).at(1)
                << "  ->  " << out.at(0)
                << "   (target " << outputs.at(i).at(0) << ")" 
                << "\n" ;
    }
}
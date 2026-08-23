#pragma once

#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cmath>


class Neuron
    {
    public:
        //Stores a vector of weights and a bias. When activated, stores inputs x for later backpropagation.
        std::vector<double> neuron_inputs;
        std::vector<double> weights;
        double bias;
        double a = 0.0; //output
        double delta = 0.0;

        Neuron(std::size_t num_inputs, std::size_t num_outputs, std::mt19937& engine) : weights(num_inputs) 
        {
          //I guess there's an optimal min/max for the distribution (Xavier uniform)
            double limit = std::sqrt(6.0/(num_inputs + num_outputs));
            std::uniform_real_distribution<double> dist{-limit,limit};

            for (auto& w : weights) w = dist(engine);
            bias = dist(engine);
        };

        //sigmoid is the old school way and in general isn't great. ReLU might be better in the future. But sigmoid ok for this.
        //static class function since it applies to all neurons.
        static double sigmoid(double x) {return 1 / (1 + std::exp(-x));}
        
        //Output a is created from forward passing the layer of neurons. For each neuron, find sigmoid(w dot x + b)
        double activate(const std::vector<double>& x)
        {
            //Store neuron inputs x for updating
            neuron_inputs = x;
            double sum = std::inner_product(weights.begin(), weights.end(), x.begin(),bias);
            a = sigmoid(sum);
            return a;
        }
        //a(1-a) used for back propagation, where a is output
        double sigmoid_derivative() const {return a*(1-a);}

        void update_weights(double delta, double learning_rate)
        {
            for (std::size_t i=0; i< weights.size(); i++)
            weights.at(i) += learning_rate * delta * neuron_inputs.at(i);
            bias += learning_rate * delta;
        }
    };

class Layer {
    public:
        std::vector<Neuron> neurons;
        std::vector<double> layer_outputs;
        std::vector<double> deltas;
        std::vector<double> prev_errors;
        Layer(std::size_t num_neurons, std::size_t num_inputs, std::size_t num_outputs, std::mt19937& engine) {
            neurons.reserve(num_neurons);
            for (std::size_t i = 0; i < num_neurons; i++)
                neurons.emplace_back(num_inputs, num_outputs, engine);
        };

        //Activates each neuron to produce an output.
        std::vector<double> forward_pass(const std::vector<double>& inputs)
        {
            //Remembering vectors have a size and capacity
            //.clear() can be used because vector capacity is same. Then emplace back a for each neuron
            layer_outputs.clear();
            for(auto& neuron: neurons)
            {
                double a = neuron.activate(inputs);
                layer_outputs.emplace_back(a);
            }
            return layer_outputs;
        }
        //Backward pass applies costs to weights. dL/dw_i = (dL/da) dot (da/dz) dot (dz/dw_i)
        // z = (w dot x + b), or sum
        // a = sigmoid(z), or output
        // For cost function L = (1/2) (a-y)^2,
        //     dL/da = a-y, da/dz = sigma'(z) = a(1-a), dz/dw_i = x_i
        // delta = (a-y)a(1-a), so dL/dw_i = delta x_i and dL/db = delta
        //Returns W^T delta, or a vector of errors
        std::vector<double> backward_pass(const std::vector<double>& errors, double learning_rate)
        {
            //assign creates capacity and sets all vector slots to zero.
            deltas.assign(neurons.size(), 0.0);

            for (std::size_t i = 0; i < neurons.size(); i++)
            {
                deltas.at(i) =  errors.at(i) * neurons.at(i).sigmoid_derivative();
            }

            const std::size_t num_inputs = neurons.at(0).weights.size();
            prev_errors.assign(num_inputs, 0.0);   

            //W^T*delta. Keep previous errors for later
            for (std::size_t i = 0; i < neurons.size(); i++)
                for (std::size_t j = 0; j < num_inputs; j++)
                    prev_errors.at(j) += neurons.at(i).weights.at(j) * deltas.at(i);

            for (std::size_t i = 0; i < neurons.size(); i++)
            {
                neurons.at(i).update_weights(deltas.at(i), learning_rate);
            }
            return prev_errors;
        };
};   

class NeuralNetwork
{
    public:
        std::vector<Layer> layers;
        double learning_rate;
        size_t epochs;

        NeuralNetwork(std::vector<Layer> layers, double learning_rate, std::size_t epochs)
        :layers(std::move(layers)), learning_rate(learning_rate), epochs(epochs) {}

        //Takes in a vector of vectors of inputs/outputs
        void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs)
        {
            for(std::size_t epoch=0; epoch < epochs; epoch++)
            {
                double totalError = 0.0;
                //For each training input vector
                for (std::size_t i = 0; i < inputs.size(); i++)
                    {
                    std::vector<std::vector<double>> activations;
                    //Reserve 1 spot for inputs for activation (initialize) then 3 other ones
                    activations.reserve(layers.size() + 1);
                    activations.push_back(inputs.at(i));
                    //Pass input through each layer. A previous layer's output is the next layers input.
                    for (std::size_t layer_no = 0; layer_no < layers.size(); layer_no++)
                        {
                        const std::vector<double>& currentInput = activations.back();
                        Layer& currentLayer = layers.at(layer_no);
                        std::vector<double> output = currentLayer.forward_pass(currentInput);
                        activations.push_back(output);
                        }
                                      
                    const std::vector<double>& prediction = activations.back();
                    const std::vector<double>& target = outputs.at(i);
                            
                    std::vector<double> outputErrors(prediction.size());
                    for (std::size_t j = 0; j < prediction.size(); j++)
                        outputErrors.at(j) = target.at(j) - prediction.at(j);  

                    totalError += std::inner_product(outputErrors.begin(), outputErrors.end(), outputErrors.begin(),0.0);

                    std::vector<double> errors = outputErrors;
                    for (std::size_t k = layers.size(); k-- > 0; )
                        errors = layers.at(k).backward_pass(errors, learning_rate);
                    }
                if (epoch % 1000 == 0)
                std::cout << "epoch " << epoch << "  mse " << totalError / inputs.size() << "\n";
            }
        }

        std::vector<double> predict(const std::vector<double>& input)
        {
            std::vector<double> a = input;
            for (auto& layer : layers)
                a = layer.forward_pass(a);
            return a;
        }
};
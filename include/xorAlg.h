#pragma once

#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cmath>


float input_min_max = 6.0/std::sqrt(2.0 + 1.0);

class Neuron
    {
    public:
        std::vector<double> weights;
        double bias;
        double output = 0.0;
        double delta = 0.0;
        std::vector<double> neuron_inputs;

        Neuron(std::size_t num_inputs, std::mt19937& engine) : weights(num_inputs) 
        {
            //I guess there's an optimal min/max for the distribution (Xavier initialization)
            std::uniform_real_distribution<double> dist{-input_min_max,input_min_max};
            for (auto& w : weights) w = dist(engine);
            bias = dist(engine);
        };

        static double sigmoid(double x) {return 1 / (1 + std::exp(-x));}
        
        //Output is created from forward passing the layer of neurons. For each neuron, find sigmoid(w dot x + b)
        double activate(const std::vector<double>& inputs)
        {
            neuron_inputs = inputs;
            double sum = std::inner_product(weights.begin(), weights.end(), inputs.begin(),bias);
            output = sigmoid(sum);
            return output;
        }
        //a(1-a) used for back propagation, where a is output
        double sigmoid_derivative() const {return output*(1-output);}

        void update_weights(double delta, double learning_rate)
        {
            for (std::size_t i=0; i< weights.size(); i++)
            weights[i] += learning_rate * delta * neuron_inputs[i];
            bias += learning_rate * delta;
        }
    };

class Layer {
    public:
        std::vector<Neuron> neurons;
        std::vector<double> layer_outputs;
        std::vector<double> deltas;
        Layer(std::size_t num_neurons, std::size_t num_inputs, std::mt19937& engine) {
            neurons.reserve(num_neurons);
            for (std::size_t i = 0; i < num_neurons; i++)
                neurons.emplace_back(num_inputs, engine);
        };

        //Activates each neuron to produce an output.
        std::vector<double> forward_pass(const std::vector<double>& inputs)
        {
            layer_outputs.clear();
            for(auto& neuron: neurons)
            {
                double output = neuron.activate(inputs);
                layer_outputs.emplace_back(output);
            }
            return layer_outputs;
        }
        //Backward pass applies costs to weights. dL/dw_i = (dL/da) dot (da/dz) dot (dz/dw_i)
        // z = (w dot x + b), or sum
        // a = sigmoid(z), or output
        // For cost function L = (1/2) (a-y)^2,
        //     dL/da = a-y, da/dz = sigma'(z) = a(1-a), dz/dw_i = x_i
        // delta = (a-y)a(1-a), so dL/dw_i = delta x_i and dL/db = delta
        // double backward_pass(const std::vector<double>& errors, double learning_rate)
        // {
        //     deltas.assign(neurons.size(), 0.0);

        //     for (std::size_t i = 0; i < neurons.size(); i++)
        //     {
        //         deltas[i] =  errors[i] * neurons[i].sigmoid_derivative();
        //         neurons[i].update_weights(deltas[i], learning_rate);
        //     }
        // };
};   
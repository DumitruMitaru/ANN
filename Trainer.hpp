#ifndef TRAINER
#define TRAINER

#include <vector>
#include <armadillo>
#include "functions.hpp"
#include "data_struct.hpp"

class Trainer
{
	public:
		float train(float step_size, std::vector<image_data>& data); 

		void set_weights_biases(std::vector<arma::fmat>* weights, std::vector<arma::fvec>* biases); 
	private:
	
		int num_layers; 
		std::vector<arma::fmat> weights_gradient; 
		std::vector<arma::fvec> biases_gradient;
		std::vector<arma::fvec> error; 
		std::vector<arma::fvec> weighted_outputs; 
		std::vector<arma::fvec> activations;
		std::vector<arma::fmat>* network_weights; 
		std::vector<arma::fvec>* network_biases; 

		void feed_forward(arma::fvec& image_data); 
		arma::fvec compute_cost_vector(arma::fvec& label_data); 
		float compute_cost(arma::fvec& label_data);
		void back_propagate(arma::fvec& cost_vector); 
		void compute_gradients(); 
		void gradient_decent(float multiplier); 
		void reset_gradients(); 
};


#endif 


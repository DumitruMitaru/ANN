#include "Trainer.hpp"



// perfor one iteration of gradient decent with set of image data and step size
void Trainer::train(float step_size, std::vector<image_data>& data)
{
	using namespace std;

	arma::vec cost_vector; 
	int num_data = data.size(); 
	float multiplier = step_size / num_data; 
	
	// iterate through image data to compute gradients
	for(int i = 0; i < num_data; ++i)
	{
		feed_forward(data[i].image);			
		cost_vector = compute_cost_vector(data[i].label);
		back_propagate(cost_vector);
		compute_gradients(); 
	}

	// perform gradient decent and then reset gradients to zero
	gradient_decent(multiplier);
	reset_gradients(); 
}



// feed training data forward and 
// compute weighted outputs and activations
// for each layer
void Trainer::feed_forward(arma::vec& image_data)
{
	activations[0] = image_data; 

	for(int i = 1; i < num_layers; ++i)
	{
		weighted_outputs[i] = network_weights->at(i) * activations[i - 1] + network_biases->at(i); 
		activations[i] = sigma(weighted_outputs[i]); 
	}
}



// compute the cost vector after training data is fed forward
arma::vec Trainer::compute_cost_vector(arma::vec& label_data)
{
	arma::vec cost_vector = activations.back() - label_data;
	
	return cost_vector; 
}



// compute cost after training data is fed forward
float Trainer::compute_cost(arma::vec& label_data)
{
	float cost = 0.5 * sum(square(label_data - activations.back())); 

	return cost; 
}



// back propagate cost vector and compute error
// for each layer
void Trainer::back_propagate(arma::vec& cost_vector)
{
	error.back() = cost_vector % dsigma(weighted_outputs.back());
	
	for(int i = num_layers - 2; i > 0; --i)
	{
		error[i] = (network_weights->at(i + 1).t() * error[i + 1]) % dsigma(weighted_outputs[i]); 
	}
}



// compute gradient for weights and biases
// for 1 training input and add them to current
// gradients. 
void Trainer::compute_gradients()
{
	// temporary gradients to be added to data member gradients
	std::vector<arma::vec> temp_gradb(num_layers); 
	std::vector<arma::mat> temp_gradw(num_layers); 

	// compute gradients
	for(int i = 1; i < num_layers; ++i)
	{	
		temp_gradb[i].copy_size(biases_gradient[i]); 
		temp_gradw[i].copy_size(weights_gradient[i]); 

		temp_gradb[i] = error[i]; 
		temp_gradw[i].each_row() = activations[i - 1].t(); 
		temp_gradw[i].each_col() %= error[i];
	}

	// add temp gradients to data member gradients
	for(int i = 1; i < num_layers; ++i)
	{
		biases_gradient[i] += temp_gradb[i]; 
		weights_gradient[i] += temp_gradw[i]; 
	}
}



// subtract gradients from weights and biases 
// using multiplier to scale gradients
void Trainer::gradient_decent(float multiplier)
{

	for(int i = 1; i < num_layers; ++i)
	{
		network_biases->at(i) -= biases_gradient[i] * multiplier;
		network_weights->at(i) -= weights_gradient[i] * multiplier;
	}

}



// give trainer access to networks weights and biases
// initialize gradients, error, activations and weighted outputs
// to proper size
void Trainer::set_weights_biases(std::vector<arma::mat>* weights, std::vector<arma::vec>* biases)
{
	num_layers = weights->size();

	network_weights = weights; 
	network_biases = biases; 

	weights_gradient.resize(num_layers); 
	biases_gradient.resize(num_layers);
	error.resize(num_layers);
	weighted_outputs.resize(num_layers);
	activations.resize(num_layers);

	for(int i = 0; i < num_layers; ++i)
	{
		weights_gradient[i].copy_size(network_weights->at(i)); 
		biases_gradient[i].copy_size(network_biases->at(i));
		error[i].copy_size(network_biases->at(i)); 
		weighted_outputs[i].copy_size(network_biases->at(i)); 
		activations[i].copy_size(network_biases->at(i)); 
	}
	reset_gradients(); 
}



// set gradients back to zero
void Trainer::reset_gradients()
{
	for(int i = 0; i < num_layers; ++i)
	{
		biases_gradient[i].zeros(); 
		weights_gradient[i].zeros();
	}
}

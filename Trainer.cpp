#include "Trainer.hpp"


float Trainer::train(float step_size, std::vector<image_data>& data)
{
	using namespace std;

	arma::fvec cost_vector; 
	int num_data = data.size(); 
	float multiplier = step_size / num_data; 
	float cost = 0; 
	for(int i = 0; i < num_data; ++i)
	{
	//	cout << "1" << endl;
		feed_forward(data[i].image);
	//	cout << "2" << endl;
		cost_vector = compute_cost_vector(data[i].label);
	//	cout << "3" << endl;
		back_propagate(cost_vector);
	//	cout << "4" << endl;
		compute_gradients();
	//	cout << "5" << endl;
//		cost += (compute_cost(data[i].label) / num_data); 
//		std::cout << compute_cost(data[i].label) << std::endl; 
	}


	gradient_decent(multiplier);
	reset_gradients(); 
//	cout << "cost is: " << cost << endl; 
//	cout << data.back().label << endl; 
//	cout << activations.back() << endl; 
//	cout << norm(weights_gradient[1]) << endl;

	return cost; 
}


// feed training data forward and 
// compute weighted outputs and activations
// for each layer
void Trainer::feed_forward(arma::fvec& image_data)
{
	activations[0] = image_data; 

	for(int i = 1; i < num_layers; ++i)
	{
		weighted_outputs[i] = network_weights->at(i) * activations[i - 1] + network_biases->at(i); 
		activations[i] = sigma(weighted_outputs[i]); 
	}
}



// compute the cost vector after training data is fed forward
arma::fvec Trainer::compute_cost_vector(arma::fvec& label_data)
{
	arma::fvec cost_vector = activations.back() - label_data;
	
	return cost_vector; 
}



// compute cost after training data is fed forward
float Trainer::compute_cost(arma::fvec& label_data)
{
	float cost = 0.5 * sum(square(label_data - activations.back())); 

	return cost; 
}



// back propagate cost vector and compute error
// for each layer
void Trainer::back_propagate(arma::fvec& cost_vector)
{
	error.back() = cost_vector % dsigma(weighted_outputs.back());
	
	for(int i = num_layers - 2; i > 0; --i)
	{
		error[i] = (network_weights->at(i + 1).t() * error[i + 1]) % dsigma(weighted_outputs[i]); 
	}
}



// compute gradient for weights and biases
// for 1 training input
void Trainer::compute_gradients()
{
	std::vector<arma::fvec> temp_gradb(num_layers); 
	std::vector<arma::fmat> temp_gradw(num_layers); 


	for(int i = 1; i < num_layers; ++i)
	{	
		temp_gradb[i].copy_size(biases_gradient[i]); 
		temp_gradw[i].copy_size(weights_gradient[i]); 

		temp_gradb[i] = error[i]; 
		temp_gradw[i].each_row() = activations[i - 1].t(); 
		temp_gradw[i].each_col() %= error[i];
	}

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
void Trainer::set_weights_biases(std::vector<arma::fmat>* weights, std::vector<arma::fvec>* biases)
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


void Trainer::reset_gradients()
{
	for(int i = 0; i < num_layers; ++i)
	{
		biases_gradient[i].zeros(); 
		weights_gradient[i].zeros();
	}
}

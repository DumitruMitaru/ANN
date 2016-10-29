#include "Network.hpp"
#include <iostream>
#include <cmath>

// function will set the size of each class variable
// based on row of integers arguements that specifies
// the number of neurons (value at index) at each layer (index number)
void Network::set_layers(arma::Row<int> new_layers)
{
	int num_layers = new_layers.size(); 
	
	layers = new_layers; 
	biases.resize(num_layers); 
	weights.resize(num_layers); 
	gradient.resize(num_layers); 
	error.resize(num_layers); 
	activs.resize(num_layers); 
	wtput.resize(num_layers); 
	
	for(int i = 1; i < num_layers; i++)
	{
		// set size of weights and biases and initialize to random values
		weights[i].set_size(layers[i],layers[i-1]);
		biases[i].set_size(layers[i]); 
		weights[i].randn(); 
		biases[i].randn(); 

		// set size of gradient
		gradient[i].set_size(layers[i],layers[i-1]);
		
		/*
		std::cout << "weights" << std::endl; 
		std::cout << weights[i] << std::endl; 
		std::cout << "Biases" << std::endl; 
		std::cout << biases[i] << std::endl; 
		*/

		// since error, activations and weighted ouputs arent defined for the input layer (layer[0])
		// dont set the size for the first element of these arrays
		if( i != 0 )
		{
			error[i].set_size(layers[i]); 
			activs[i].set_size(layers[i]); 
			wtput[i].set_size(layers[i]); 
		}
	}
	
}



// feed forward inputs throug network and calculate
// weighted ouputs and activations for each layer
void Network::feed_forward(arma::fvec inputs)
{
	// set first layer of activations to inputs
	activs[0] = inputs;
	int num_layers = layers.size(); 

	for(int i = 1; i < num_layers; ++i)
	{
		wtput[i] = weights[i]*activs[i-1] + biases[i]; 
		activs[i] = vec_sigma(wtput[i]); 
		/*
		std::cout<< "activations of:  " << i << " layer" << std::endl; 
		std::cout << activs[i] << std::endl; 
		*/
	}
}



// 
void Network::back_propagate()
{
	int num_layers = layers.size() - 1;  

	for(int i = num_layers; i > 0; i++)
	{
		
	}
}


// compute the error of the last layer in the network
void Network::output_error(arma::fvec training_value)
{
	error.back() = this->get_cost_gradient(training_value) % vec_dsigma(wtput.back()); 
}


// compute the cost function from the output
// of the nueral network and the true value
// this cost function uses quadratic cost
float Network::get_cost(arma::fvec true_value)
{
	return 0.5 * pow(norm(this->get_cost_gradient), 2); 
}



// compute cost vector for single training input
// and return result
arma::fvec get_cost_gradient(arma::fvec train_value)
{
	arma::fvec cost_gradient; 
	
	cost_gradient = activs.back()- train_value; 

	return cost_gradient; 
}


// compute the sigma function on a vector of values
// and return result as vector
arma::fvec Network::vec_sigma(arma::fvec input_vector)
{	
	// get size of input vector and 
	// initialize vector activations vector
	// of same size
	int size = input_vector.size(); 
	arma::fvec activations(input_vector.size()); 

	// compute the sigma function on each value
	// in input vector and store in 
	// activation vector
	for(int i = 0; i < size; ++i)
	{
		activations[i] = sigma(input_vector[i]); 
	}

	return activations; 
}


// compute the derivative of the sigma function
// with a vector of float values and return
// the result as a vector 
arma::fvec Network::vec_dsigma(arma::fvec input_vector)
{
	// get size of input vector and
	// initialze a derivative vecotor
	// of same size
	int size = input_vector.size(); 
	arma::fvec deriv(size);

	// compute the derivative of each element
	// in the input vector as store in 
	// derivative vecotor
	for(int i = 0; i < size; i++)
	{
		deriv[i] = dsigma(input_vector[i]); 
	}

	return deriv;
}



float Network::sigma(float value)
{
	float exp = pow(2.71827, -value); 
	return 1.0 / (1.0 + exp); 
}



float Network::dsigma(float value)
{
	return sigma(value)*(1 - sigma(value)); 
}



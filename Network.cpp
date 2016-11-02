#include "Network.hpp"
#include <iostream>
#include <cmath>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>


// based on row of integers arguements that specifies
// the number of neurons (value at index) at each layer (index number)
void Network::set_layers(arma::Row<int> new_layers)
{
	num_layers = new_layers.size(); 	
	layers = new_layers; 
	weights.resize(num_layers); 
	biases.resize(num_layers);	
	for(int i = 1; i < num_layers; i++)
	{
		// set size of weights and biases and initialize to random values
		weights[i].set_size(layers[i],layers[i-1]);
		biases[i].set_size(layers[i]); 
		weights[i].randn(); 
		biases[i].randn(); 

	}
	
}



// perform gradient decent with a sample of training inputs and output and step size
void Network::gradient_decent(float step_size, std::vector<arma::fvec>& training_inputs, std::vector<arma::fvec>& training_outputs)
{
	using namespace std; 
	// initialize vectors to hold gradients, error, activations, weighted outputs
	// and a temp vector to hold gradient of weights. 
	std::vector<arma::fmat> weights_gradient(num_layers); 
	std::vector<arma::fvec> biases_gradient(num_layers); 
	std::vector<arma::fvec> error(num_layers); 
	std::vector<arma::fvec> activations(num_layers); 
	std::vector<arma::fvec> weighted_outputs(num_layers); 
	std::vector<arma::fmat> temp_grad(num_layers); 
	
	// intialize cost and partial cost...used to display to user. 
	float cost = 0; 
	float partial_cost = 0; 	
	
	// initialize number of inputs to train on and multiplier
	int num_trainers = training_inputs.size(); 
	float multiplier = -1 * step_size / num_trainers; 

	// initialize gradients to size of current network
	for(int i = 1; i < num_layers; ++i)
	{
		weights_gradient[i].copy_size(weights[i]); 
		biases_gradient[i].copy_size(biases[i]); 
	}

	// feed forward training input, back propagate and compute gradients
	for(int i = 0; i < num_trainers; ++i)
	{	

		feed_forward(training_inputs[i], weighted_outputs, activations);
		error = back_propagate(training_outputs[i], weighted_outputs, activations.back()); 
		temp_grad = compute_partial_gradient_weights(activations, error);
		
		// sum gradients for all training samples
		for(int j = 0; j < num_layers; ++j)
		{
			weights_gradient[j] += temp_grad[j]; 
			biases_gradient[j] += error[j]; 
		}
		
		// compute cost value of each training data
		// this is mainly to output to the user..it is not used in any calculation
		partial_cost = compute_partial_cost(training_outputs[i], activations.back());
//		cout << partial_cost << " "; 
		cost += partial_cost / num_trainers; 
	}
		
//		std::cout << "the cost is:  " << cost << "  ";

	// increment weights and biases by gradients times multiplier
	for(int i = 1; i < num_layers; ++i)
	{
		weights[i] = weights[i] + (weights_gradient[i] * multiplier); 
		biases[i] = biases[i] + (biases_gradient[i] * multiplier); 
	}
//	cout << "grad is:  "  << norm(weights_gradient[2]) << "  "; 
//	cout << "weights is :  " << norm(weights[2]) << endl; 

}



// feed forward inputs throug network and calculate
// weighted ouputs and activations for each layer
void Network::feed_forward(arma::fvec& training_input, std::vector<arma::fvec>& weighted_output, std::vector<arma::fvec>& activations)
{
	// set first layer of activations to inputs
	activations[0] = training_input; 

	for(int i = 1; i < num_layers; ++i)
	{
		weighted_output[i] = weights[i]*activations[i-1] + biases[i]; 
		activations[i] = vec_sigma(weighted_output[i]); 
	}
}



// back propagate the error in the last layer
// through the network
std::vector<arma::fvec> Network::back_propagate(arma::fvec& training_output, std::vector<arma::fvec>& weighted_output, arma::fvec& network_output)
{	

	// initialize error as vector of column vectors of floats
	std::vector<arma::fvec> error(num_layers);  

	// output error for last layer
	error.back() = compute_partial_cost_gradient(training_output, network_output) % vec_dsigma(weighted_output.back()); 

	// begin at second to last layer and back propagate
	// to second layer since first layer is input layer
	for(int i = num_layers - 2; i > 0; --i)
	{
		error[i] = (weights[i+1].t() * error[i+1]) % vec_dsigma(weighted_output[i]); 
	}

	return error; 
}



// compute the partial gradient for weights
std::vector<arma::fmat> Network::compute_partial_gradient_weights(std::vector<arma::fvec>& activations, std::vector<arma::fvec>& error )
{
	// compute gradient for weights
	std::vector<arma::fmat> weights_gradient(num_layers); 

	for(int i = 1; i < num_layers; ++i)
	{
		weights_gradient[i].copy_size(weights[i]); 
		weights_gradient[i].each_row() = activations[i-1].t(); 
		weights_gradient[i].each_col() %= error[i]; 
	}

	return weights_gradient; 
}


// compute the cost function from the output
// of the nueral network and the true value
// this cost function uses quadratic cost
float Network::compute_partial_cost(arma::fvec& training_output, arma::fvec& network_output)
{	
	//using namespace std; 
	//cout << training_output.t() << endl; 
	//cout << network_output.t() << endl; 

	return 0.5 * sum(square((training_output -  network_output))); 
}


// compute cost vector for single training input
// and return result
arma::fvec Network::compute_partial_cost_gradient(arma::fvec& training_output,arma::fvec& network_output)
{
	arma::fvec cost_gradient; 
	
	cost_gradient = network_output - training_output; 

	return cost_gradient; 
}



// save weights and biases
void Network::save()
{
	using namespace std; 
	// declare output stream file
	// string to be appended to file name to indicate layer number
	// temp string to hold file name with layer number
	ofstream layer_file; 
	char layer[] = "2";  
	char temp[20]; 
	
	// remove old neural network file to store number of layers
	remove_files(); 
	layer_file.open(LAYER_FILE); 

	if(layer_file)
		layer_file << num_layers << endl; 
	else 
		cout << "layer file not opened" << endl; 
	layer_file.close();

	// store weights and biases as WEIGHTS_FILE + layer
	for(int i = 1; i < num_layers; ++i)
	{
		strcpy(temp, WEIGHTS_FILE); 
		weights[i].save(strcat(temp, layer), arma::arma_ascii);
		strcpy(temp, BIASES_FILE);
		biases[i].save(strcat(temp, layer), arma::arma_ascii);
		++(*layer); 
	}
}



// load weights and biases
void Network::load()
{
	using namespace std; 
	// declare input stream to read number of layers
	// string to indicate layer number to read from
	// and temp string to hold file name to read from
	ifstream layer_input; 
	char layer[] = "2"; 
	char temp[20]; 

	// open file and read number of layers into num_layers
	layer_input.open(LAYER_FILE); 
	layer_input >> num_layers; 
	
	// resize weights and biases
	weights.resize(num_layers); 
	biases.resize(num_layers); 
	
	// read in weights and baises
	for(int i = 1; i < num_layers; ++i)
	{
		strcpy(temp, WEIGHTS_FILE); 
		strcat(temp, layer); 
		weights[i].load(temp); 
		strcpy(temp, BIASES_FILE); 
		strcat(temp, layer); 
		biases[i].load(temp); 
		++(*layer); 
	}
}

// removes biases, weights and layer files previously stored. 
void Network::remove_files()
{
	char layer[] = "1"; 
	char temp[20]; 

	remove(LAYER_FILE); 
	
	do
	{
		++(*layer);
		strcpy(temp, BIASES_FILE); 
		strcat(temp, layer); 
		remove(temp); 
		strcpy(temp, WEIGHTS_FILE);
		strcat(temp, layer); 

	}while(remove(temp) == 0);
}



// compute the sigma function on a vector of values
// and return result as vector
arma::fvec Network::vec_sigma(arma::fvec& input_vector)
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
arma::fvec Network::vec_dsigma(arma::fvec& input_vector)
{
	// get size of input vector and
	// initialze a derivative vecotor
	// of same size
	int size = input_vector.size(); 
	arma::fvec deriv(size);

	// compute the derivative of each element
	// in the input vector as store in 
	// derivative vector
	for(int i = 0; i < size; i++)
	{
		deriv[i] = dsigma(input_vector[i]); 
	}

	return deriv;
}


// sigma function
float Network::sigma(float& value)
{
	float exp = pow(2.71827, -value); 
	return 1.0 / (1.0 + exp); 
}


// derivative of sigma function
float Network::dsigma(float& value)
{
	return sigma(value)*(1 - sigma(value)); 
}



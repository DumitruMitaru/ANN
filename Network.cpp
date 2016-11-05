#include "Network.hpp"
#include "functions.hpp"
#include <iostream>
#include <cmath>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include "Trainer.hpp"
#include "data_struct.hpp"


void Network::train(float step_size, std::vector<image_data>& train_data, int iterations)
{
	for(int i = 0; i < iterations; ++i)
		 teacher.train(step_size, train_data); 
}

float Network::test(std::vector<image_data>& test_data)
{
	int true_value; 
	int network_output; 
	float num_correct = 0;
	float perc_correct;
	int num_data = test_data.size(); 

	for(int i = 0; i < num_data; ++i)
	{
		network_output = compute(test_data[i].image); 
		true_value = test_data[i].label.index_max();
//		std::cout << test_data[i].label << std::endl; 

		if(network_output == true_value)
			++num_correct; 
	}

	perc_correct = 100.0 * num_correct / num_data; 
	return perc_correct; 
}


int Network::compute(arma::fvec& input)
{
	std::vector<arma::fvec> activations(num_layers);
	std::vector<arma::fvec> weighted_outputs(num_layers); 
	activations[0] = input; 

	for(int i = 1; i < num_layers; ++i)
	{	
		weighted_outputs[i] = weights[i] * activations[i - 1] + biases[i];
		activations[i] = sigma(weighted_outputs[i]); 
	}

//		std::cout << activations.back() << std::endl; 
	return activations.back().index_max(); 
}

// based on row of integers arguements that specifies
// the number of neurons (value at index) at each layer (index number)
void Network::set_layers(arma::Row<int> new_layers)
{
	arma::Row<int> lay = new_layers; 
	num_layers = lay.size(); 	
	weights.resize(num_layers); 
	biases.resize(num_layers);	

	for(int i = 1; i < num_layers; i++)
	{
		// set size of weights and biases and initialize to random values
		weights[i].set_size(lay[i],lay[i-1]);
		biases[i].set_size(lay[i]); 
		weights[i].randn();
		biases[i].randn();

	}

	teacher.set_weights_biases(&weights, &biases); 
	
}




// save weights and biases
void Network::save()
{
	using namespace std; 
	// declare output stream file
	// string to be appended to file name to indicate layer number
	// temp string to hold file name with layer number
	ofstream layer_file; 
	int layer = 2;
	string temp; 
	
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
		temp = WEIGHTS_FILE; 
		temp = temp + to_string(layer);
		weights[i].save(temp.c_str(), arma::arma_ascii);
		temp = BIASES_FILE; 
		temp = temp + to_string(layer);
		biases[i].save(temp.c_str(), arma::arma_ascii);
		++layer; 
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
	int layer = 2;  
	string temp; 

	// open file and read number of layers into num_layers
	layer_input.open(LAYER_FILE); 
	layer_input >> num_layers; 
	
	// resize weights and biases
	weights.resize(num_layers); 
	biases.resize(num_layers); 
	
	// read in weights and baises for how many layers there are
	// and disregard the first layer since it is the input layer
	for(int i = 1; i < num_layers; ++i)
	{
		temp = WEIGHTS_FILE; 
		temp = temp + to_string(layer);
		weights[i].load(temp.c_str(), arma::arma_ascii); 
		temp = BIASES_FILE; 
		temp = temp + to_string(layer);
		biases[i].load(temp.c_str(), arma::arma_ascii); 
		++layer; 
	}

	teacher.set_weights_biases(&weights, &biases); 

}

// removes biases, weights and layer files previously stored. 
void Network::remove_files()
{	
	using namespace std;
	int layer = 2; 
	string temp; 

	remove(LAYER_FILE); 
	
	do
	{
		temp = BIASES_FILE;
		temp = temp + to_string(layer);
		remove(temp.c_str()); 
		temp = WEIGHTS_FILE; 
		temp = temp + to_string(layer);
		++layer; 
	}while(remove(temp.c_str()) == 0);
}



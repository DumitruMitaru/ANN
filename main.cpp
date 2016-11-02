#include <iostream>
#include <vector>
#include <fstream>
#include <armadillo>
#include <cstdint>
#include "Network.hpp"
#include "Images.hpp"
#include <cstdlib>
#include <ctime>
#include <stdio.h>

using namespace std; 
using namespace arma;

int main()
{
	
	Row<int> lay; 				// vector to hold layers and number of neurons in each layer
	Images trainer; 			// object to load a feel image data
	Network brain; 				// network
	vector<training_data> sample;		// vector of image data in vector of struct...not used for network functions so I should fix this
	vector<fvec> training_inputs(10); 	// vector of training image data
	vector<fvec> training_outputs(10); 	// vector of training label data
	char user_input; 			// user input from command line

	// ask if user would like to load a previously saved network
	cout << "load weights and biases? y or n:  "; 
	cin >> user_input; 
	if(user_input == 'y')
		brain.load(); 
	else
	{
		lay << 784 << 30 << 10; 	// create a network with 28 * 28 = 784 neruons in the first layer and 30 and 10 in the second. 
		brain.set_layers(lay); 		// set layers
	}
	//load training data
	trainer.read_store_data(); 

	// iterate through training samples and train network`
	for(int k = 0; k < 10; ++k)
	{
		// get sample of data from trainer and convert to vector of fvecs
		sample = trainer.get_random_data(10); 
		for(int i = 0; i < 10; ++i)
		{
			training_inputs[i] = sample[i].image; 
			training_outputs[i] = sample[i].label; 
		}

		
//		cout << "new set" << endl; 	
		// perform 10 steps of gradient decent with this data sample
		for(int j = 0; j < 10; ++j)
		{
			brain.gradient_decent(0.005, training_inputs, training_outputs); 
		}	
	}

	// ask if user would like to save data
	cout << "save weights and biases? y or n:  "; 
	cin >> user_input; 
	if(user_input == 'y')
		brain.save(); 

	return 0; 
}

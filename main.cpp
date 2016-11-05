#include <iostream>
#include <vector>
#include <fstream>
#include <armadillo>
#include <cstdint>
#include "Network.hpp"
#include "Data_manager.hpp"
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <string>
#include "data_struct.hpp"

using namespace std; 
using namespace arma;

int main()
{
	
	Row<int> lay; 				// vector to hold layers and number of neurons in each layer
	Data_manager data; 			// object to load a feel image data
	Network brain; 				// network
	vector<image_data> sample;		// vector of image data in vector of struct...not used for network functions so I should fix this
//	vector<image_data> test; 
	char user_input; 			// user input from command line
	int iter = 0; 				// number of training iterations.
	float init_acc = 0; 
	float final_acc = 0; 

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
	
	cout << "do you want to train? y or n: "; 
	cin >> user_input; 
	cin.ignore(100, '\n'); 

	data.read_store_data(); 

	if(user_input == 'y')
	{
		cout << "number of iterations:  "; 
		cin >> iter; 
		cin.ignore(100, '\n');

		
		sample = data.get_random_data(1000);
		init_acc = brain.test(sample);

		for(int i = 0; i < iter; ++i)
		{
			sample = data.get_random_data(100);	
			brain.train(1, sample, 1);
//			sample = data.get_random_data(1000); 
//			cout << "percent accuracy: %" << brain.test(sample) << endl; 

		}

		sample = data.get_random_data(1000); 
		final_acc = brain.test(sample); 
		
		cout << "ACCURACY BEFORE TRAINING: %" << init_acc << endl; 
		cout << "ACCURACY AFTER TRAINING: %" << final_acc << endl; 
	}
	else
	{
		do
		{
			sample = data.get_random_data(1); 
			cout << "true value: " << sample[0].label.index_max() << endl; 
			cout << "network output: " << brain.compute(sample[0].image) << endl; 
			cout << endl; 
			cout << "continue? y or n: "; 
			cin >> user_input; 
			cin.ignore(100, '\n');
			cout << endl; 

		}while(user_input == 'y'); 
	}

	// ask if user would like to save data
	cout << "save weights and biases? y or n:  "; 
	cin >> user_input; 
	if(user_input == 'y')
		brain.save(); 

	return 0; 
}

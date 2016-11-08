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


// use to hold hyper parameters for optimizing neural network
struct parameters
{
	int sample_size; 
	float step_size; 
	int epochs; 
	int updates; 
};



void optimize(Network& ann, std::vector<parameters> list, int list_size); // iterate through list of parameters and train network, output and save data
void print_results(parameters& set, float accuracy, const char file_name[]);	  // print hyper parameters and accuracy of network obtained from said parameters
parameters create_parameters(int sample_size, float step_size, int epochs, int updates);	// create parameter structure
void save(parameters& set, float accuracy);	// save hyper parameters and accuracy of network obtained form said parameters


// short program for how the network can be trained
int main()
{
	
	Row<int> lay; 				// vector to hold layers and number of neurons in each layer
	Network brain; 				// network
	vector<parameters> list;		// list of hyper parameters
	
	lay << 784 << 30 << 10; 	// create a network with 28 * 28 = 784 neruons in the first layer and 30 and 10 in the second. 
	brain.set_layers(lay); 		// set layers
	

	
	// list of hyper parameter to test for
	list.push_back(create_parameters(10, 0.03, 20, 1));
	list.push_back(create_parameters(10, 0.3, 20, 1)); 
	list.push_back(create_parameters(10, 0.8, 20, 1)); 
	list.push_back(create_parameters(10, 1.0, 20, 1)); 
	list.push_back(create_parameters(10, 3.0, 20, 1));
//	list.push_back(create_parameters(100, 2, 100, 100)); 
//	list.push_back(create_parameters(10, 3, 30, 100)); 

	
	// train network with list of parameters	
	optimize(brain, list, list.size()); 
		
	return 0; 
}

void optimize(Network& ann, std::vector<parameters> list, int list_size)
{
	using namespace std;

	float accuracy;			// evaluated after each epoch
	Data_manager manager;		// used to retrieve test and training data
	std::vector<std::vector<image_data> > train_data; 	// set of disjoint subsets of total training data
	std::vector<image_data> test_data; 			// set of testing data
	int num_mini_batches; 					// number of mini_batches in training data
	char file_name = "testing results.txt"; 		// file to save results to

	manager.load();			     // load training and test data 
	test_data = manager.get_test_data(); // retrieve test data

	// iterate through the list of hyper parameters and test network
	for(int i = 0; i < list_size; ++i)
	{
		train_data = manager.get_mini_batches(list[i].sample_size);  // obtain training data in mini batches of sample size
		num_mini_batches = train_data.size();			     // calc number of batches

		// iterate through epochs
		for(int j = 0; j < list[i].epochs; ++j)
		{
			// iterate through training data and perform SGD with each subset
			for(int k = 0; k < num_mini_batches; ++k)
			{
				ann.train(list[i].step_size, train_data[k], list[i].updates);
			}

			// calc and print accuracy
			cout << "epoch " << j << " complete: "; 
			accuracy = ann.test(test_data); 
			cout << "accuracy %" << accuracy << endl; 

		}
		
		// print results of training with set of hyper paramters, save and reset network weights and biases
		print_results(list[i], accuracy);
		save(list[i], accuracy, file_name); 
		ann.reset(); 
	}


}



parameters create_parameters(int sample_size, float step_size, int epochs, int updates)
{
	parameters set_to_return; 
	
	set_to_return.sample_size = sample_size; 
	set_to_return.step_size = step_size; 
	set_to_return.epochs = epochs; 
	set_to_return.updates = updates; 

	return set_to_return; 
}



void print_results(parameters& set, float accuracy)
{
	cout << endl;
	cout << "sample size: " << set.sample_size << endl; 
	cout << "step size:  " << set.step_size << endl; 
	cout << "number of epochs:  " << set.epochs << endl; 
	cout << "updates per epoch: " << set.updates << endl; 
	cout << "accuracy: " << accuracy << endl; 
	cout << endl; 
}

void save(parameters& set, float accuracy, const char file_name)
{
	ofstream data_file; 

	data_file.open(file_name, ios::app); 

	if(data_file)
	{
		data_file << endl; 
		data_file << "sample size: " << set.sample_size << endl;
		data_file << "step size: " << set.step_size << endl;
		data_file << "number of epochs: " << set.epochs << endl;
		data_file << "updates per epoch: " << set.updates << endl; 
		data_file << "accuracy: " << accuracy << endl;

		data_file.close(); 
	}

}

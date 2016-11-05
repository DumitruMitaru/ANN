#include <vector>
#include <armadillo>
#include "Trainer.hpp"
#include "data_struct.hpp"

const char LAYER_FILE[] = "layers.txt"; 
const char WEIGHTS_FILE[] = "weights_at_layer_"; 
const char BIASES_FILE[] = "biases_at_layer_"; 

class Network
{
	public: 
		// each element in new_layers specifies the number of neurons in the layer at element index and initialize to random value
		void set_layers(arma::Row<int> new_layers); 
		void train(float step_size, std::vector<image_data>& train_data, int iterations); 
		float test(std::vector<image_data>& test_data); 
		int compute(arma::fvec& input); 
		void save(); 					// save Neural Network with weights and biases 
		void load(); 					// load Neural Network
	private:
		int num_layers; 				// number of layers in neural network...input layer counts as a layer
		std::vector<arma::fmat> weights;  	 	// vector of matrices to hold weights of each layer
		std::vector<arma::fvec> biases;                 // vector of vecotr of floats to hold biases of each layer
		Trainer teacher; 

		void remove_files(); 					// remove old neural network files...used before new data is saved
};



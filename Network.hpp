#include <vector>
#include <armadillo>
#include "Trainer.hpp"
#include "data_struct.hpp"

const char LAYER_FILE[] = "layers.txt"; 		// file to store how many layers the network has
const char WEIGHTS_FILE[] = "weights_at_layer_"; 	// prefix of file to store weights at given  layer
const char BIASES_FILE[] = "biases_at_layer_"; 		// prefix of file to store biases at given layer

class Network
{
	public: 
		void set_layers(arma::Row<int> new_layers); // set network structure..1st layer in input layer
		void train(float step_size, std::vector<image_data>& train_data, int updates); // train network
		float test(std::vector<image_data>& test_data); // test network against a set of images and return accuracy
		int compute(arma::vec& input); 			// compute image input and return value
		void save(); 					// save Neural Network weights and biases 
		void load(); 					// load previously saved Neural Network
		void reset(); 					// reset weights and biases to random; 
	private:
		int num_layers; 				// number of layers in neural network...input layer counts as a layer
		std::vector<arma::mat> weights;  	 	// vector of matrices to hold weights of each layer
		std::vector<arma::vec> biases;                 // vector of vecotr of floats to hold biases of each layer
		Trainer teacher; 				// teacher will use  SGD algorithms to train network.

		void remove_files(); 				// remove old neural network files in current dir...used before new data is saved
};



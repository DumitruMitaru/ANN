#include <vector>
#include <armadillo>


const char LAYER_FILE[] = "layers.txt"; 
const char WEIGHTS_FILE[] = "weights_at_layer_"; 
const char BIASES_FILE[] = "biases_at_layer_"; 

class Network
{
	public: 
		// each element in new_layers specifies the number of neurons in the layer at element index and initialize to random value
		void set_layers(arma::Row<int> new_layers); 
		// perform 1 iteration of gradient decent with step size (0.01) training inputs and training outputs
		void gradient_decent(float step_size, std::vector<arma::fvec>& training_inputs, std::vector<arma::fvec>& training_outputs); 
		void save(); 					// save Neural Network with weights and biases 
		void load(); 					// load Neural Network
	private:
		int num_layers; 				// number of layers in neural network...input layer counts as a layer
		arma::Row<int> layers;  		        // vector of ints to hold how many neurons are in each layer
		std::vector<arma::fmat> weights;  	 	// vector of matrices to hold weights of each layer
		std::vector<arma::fvec> biases;                 // vector of vecotr of floats to hold biases of each layer
		
		// feed input forward and calculate and store weighted inputs and activations of each layer
		void feed_forward(arma::fvec& training_input, std::vector<arma::fvec>& weighted_output, std::vector<arma::fvec>& activations);
		// back propagate error through the network and return error of each layer
		std::vector<arma::fvec> back_propagate(arma::fvec& training_output, std::vector<arma::fvec>& weighted_output, arma::fvec& network_output);
		// compute and return the gradient for one training sample. 
		std::vector<arma::fmat> compute_partial_gradient_weights(std::vector<arma::fvec>& activations, std::vector<arma::fvec>& error);
		// compute the cost of 1 training sample
		float compute_partial_cost(arma::fvec& training_output, arma::fvec& network_output);  
		  // compute and return the gradient of the cost function with respect to the activations of the last layer	
		arma::fvec compute_partial_cost_gradient(arma::fvec& training_output, arma::fvec& network_output); 		
	
		void remove_files(); 					// remove old neural network files...used before new data is saved
		float sigma(float& value);				// sigma function
		float dsigma(float& value);				// derivative of sigma function
		arma::fvec vec_sigma(arma::fvec& z);			// sigma function with vecotors as arguments and return values
		arma::fvec vec_dsigma(arma::fvec& z); 			// derivative of sigma function with vectors as arguments and return values
}; 


#include <vector>
#include <armadillo>
#include "functions.hpp"
#include "data_struct.hpp"


// class will implement SGD functions to train network. 
class Trainer
{
	public:
		void train(float step_size, std::vector<image_data>& data); // perform single step of SGD with set of image data and step size
		void set_weights_biases(std::vector<arma::mat>* weights, std::vector<arma::vec>* biases); // give trainer access to current networks wieghts and biases
	private:
	
		int num_layers; 				// number of layers in network
		std::vector<arma::mat> weights_gradient; 	// gradient of weights..used to sum up partial gradients 
		std::vector<arma::vec> biases_gradient;		// gradient of biases..used to sum up partial gradients
		std::vector<arma::vec> error; 			// error computed in back propagate function 
		std::vector<arma::vec> weighted_outputs; 	// weighted outputs computed in feed forward 
		std::vector<arma::vec> activations;		// activations computed in feed forward
		std::vector<arma::mat>* network_weights; 	// pointer to network weights
		std::vector<arma::vec>* network_biases; 	// pointer to network biases

		void feed_forward(arma::vec& image_data);             // feed inputs forward, compute activations and weighted outputs
		arma::vec compute_cost_vector(arma::vec& label_data); // compute cost vector
		float compute_cost(arma::vec& label_data);	      // compute and return cost from cost vector
		void back_propagate(arma::vec& cost_vector); 	      // back propagate cost vecotor
		void compute_gradients(); 			      // compute gradients 
		void gradient_decent(float multiplier); 	      // perform one step of gradient decent
		void reset_gradients(); 			      // reset gradients for next SGD iteration
};




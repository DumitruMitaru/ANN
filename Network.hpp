#include <vector>
#include <armadillo>

class Network
{
	public: 

	void set_layers(arma::Row<int> new_layers); 	// each element in new_layers specifies the number of neurons in the layer at element index
	void feed_forward(arma::fvec inputs);
	private: 
	
	arma::Row<int> layers;  	   // vector of ints to hold how many neurons are in each layer
	std::vector<arma::fvec> biases;    // vector of column vectors to hold biases of each layer
	std::vector<arma::fmat> weights;   // vector of matrices to hold weights of each layer
	std::vector<arma::fmat> gradient;  // vector of matrices to hold gradient of weights and biases
	std::vector<arma::fvec> error; 	   // vector of column vectors to hold errors of each layer
	std::vector<arma::fvec> activs;    // vector of column vectors to hold activations of each layer
	std::vector<arma::fvec> wtput;	   // vector of column vectors to hold weighted inputs of each layer
	
	
	void back_propagate();
	
	void output_error(arma::fvec training_value);
	float get_cost(arma::fvec train_value);  
	arma::fvec get_cost_gradient(arma::fvec train_value);   // compute and return the gradient of the cost function with respect to the activations of the last layer	
	float sigma(float value);				// sigma function
	float dsigma(float value);				// derivative of sigma function
	arma::fvec vec_sigma(arma::fvec z);			// sigma function with vecotors as arguments and return values
	arma::fvec vec_dsigma(arma::fvec z); 			// derivative of sigma function with vectors as arguments and return values
}; 


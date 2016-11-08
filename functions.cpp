#include "functions.hpp"



// sigma function
double sigma(double& value)
{
	double exp = pow(2.71827, -value); 
	return 1.0 / (1.0 + exp); 
}



// derivative of sigma function
double dsigma(double& value)
{
	return sigma(value)*(1 - sigma(value)); 
}



// compute the sigma function on a vector of values
// and return result as vector
arma::vec sigma(arma::vec& input_vector)
{	
	using namespace arma; 	
	vec out = pow((exp(-1*input_vector) + 1), -1); 
	return out; 
}



// compute the derivative of the sigma function
// with a vector of double values and return
// the result as a vector 
arma::vec dsigma(arma::vec& input_vector)
{
	using namespace arma; 
	vec temp = 1 - input_vector; 
	vec out = sigma(input_vector) % sigma(temp);  
	return out; 
}


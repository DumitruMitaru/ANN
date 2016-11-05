#include "functions.hpp"



// sigma function
float sigma(float& value)
{
	float exp = pow(2.71827, -value); 
	return 1.0 / (1.0 + exp); 
}



// derivative of sigma function
float dsigma(float& value)
{
	return sigma(value)*(1 - sigma(value)); 
}



// compute the sigma function on a vector of values
// and return result as vector
arma::fvec sigma(arma::fvec& input_vector)
{	
	using namespace arma; 	
	fvec out = pow((exp(-1*input_vector) + 1), -1); 
	return out; 
}



// compute the derivative of the sigma function
// with a vector of float values and return
// the result as a vector 
arma::fvec dsigma(arma::fvec& input_vector)
{
	using namespace arma; 
	fvec temp = 1 - input_vector; 
	fvec out = sigma(input_vector) % sigma(temp);  
	return out; 
}


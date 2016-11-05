#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <armadillo>

float sigma(float& value);
float dsigma(float& value);
arma::fvec sigma(arma::fvec& input_vector);
arma::fvec dsigma(arma::fvec& input_vector);

#endif

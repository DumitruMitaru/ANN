#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <armadillo>

double sigma(double& value);
double dsigma(double& value);
arma::vec sigma(arma::vec& input_vector);
arma::vec dsigma(arma::vec& input_vector);

#endif

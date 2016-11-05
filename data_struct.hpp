#ifndef IMAGE_DATA_HPP 
#define IMAGE_DATA_HPP

#include <armadillo>

// structure to hold image data and label data
// from MNIST data set
struct image_data
{
	arma::fvec image; 
	arma::fvec label; 
};

#endif

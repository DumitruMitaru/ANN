#include <armadillo>
#include <vector>
#include <fstream>


const int NUM_IMAGES = 60000; // number of images in training data set
const int NUM_PXLS = 28 * 28; // pixel size

const char IMAGE_FILE[] = "train-images.idx3-ubyte"; // image data file 
const char LABEL_FILE[] = "train-labels.idx1-ubyte"; // image label data file

// structure to pair image data with label
struct training_data
{
	arma::fvec image; 
	arma::fvec label; 
};

// class will load image and label data and provide them to network for training and testing
class Images
{
	public: 

		std::vector<training_data> get_random_data(int num_data);  // return a vector of random data to train/test
		void read_store_data(); 				   // load data from files. 
	
	private: 
		std::vector<training_data> data;			   // variable to hold image and label data
	
};

bool is_in(int* array, int size, int number);  // is integer in array of size?

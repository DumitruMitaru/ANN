#include <armadillo>
#include <vector>
#include <fstream>
#include "data_struct.hpp"

const int NUM_TRAIN_IMAGES = 60000; // number of images to train with
const int NUM_TEST_IMAGES = 10000;  // number of images to test with
const int NUM_PXLS = 28 * 28; // pixel size

const char TRAIN_IMAGE_FILE[] = "train-images.idx3-ubyte"; // training image data file 
const char TRAIN_LABEL_FILE[] = "train-labels.idx1-ubyte"; // training image label data file
const char TEST_IMAGE_FILE[] = "t10k-images.idx3-ubyte";   // testing image data file
const char TEST_LABEL_FILE[] = "t10k-labels.idx1-ubyte";   // testing image label data file

// class will load image and label data and provide them to network for training and testing
class Data_manager
{
	public: 

		std::vector<image_data> get_random_data(int num_data);  // return a vector of random data to train/test
		std::vector<std::vector<image_data> >  get_mini_batches(int mini_batch_size); // return partioned train_data into batches of mini_batch_size
		std::vector<image_data> get_test_data(); 
		void load(); 				   // load data from files. 

	private: 
		std::vector<image_data> train_data;	   // used to hold image and label data for training
		std::vector<image_data> test_data; 	   // used to hold image and label data for testing

		void load_train_data(); 
		void load_test_data(); 
	
};

bool is_in(int* array, int size, int number);  // is number in array of size?...used for returning unique sets of random data

#include <armadillo>
#include <vector>
#include <fstream>


const int NUM_IMAGES = 60000; 
const int NUM_PXLS = 28 * 28; 

const char IMAGE_FILE[] = "train-images.idx3-ubyte"; 
const char LABEL_FILE[] = "train-labels.idx1-ubyte";

struct training_data
{
	arma::fvec image; 
	arma::fvec label; 
};
class Images
{
	public: 
		std::vector<training_data> get_random_data(int num_data); 
		void read_store_data(); 
	private: 
		std::vector<training_data> data; 
	
};

bool is_in(int* array, int size, int number); 

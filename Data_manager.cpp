#include "Data_manager.hpp"
#include <cstdlib>
#include <ctime>


// read and store training images and labels. 
void Data_manager::read_store_data()
{
	using namespace std; 
	ifstream images; 
	ifstream labels; 
	int label_val = 0; 
	images.open(IMAGE_FILE); 
	labels.open(LABEL_FILE); 
	
	data.resize(NUM_IMAGES); 

	if(images && labels)
	{
		// ignore first number of bytes of each file
		for(int i = 0; i < 16; ++i)
			images.get(); 
		for(int i = 0; i < 8; ++i)
			labels.get(); 

		for(int i = 0; i < NUM_IMAGES; ++i)
		{
			// read in label data and set a vector of 10 elements to zero
			// setting the index value to the label value and setting that element 
			// at that index value to 1 
			data[i].label.set_size(10); 
			data[i].label.zeros();
			label_val = labels.get(); 
			data[i].label[label_val] = 1; 
			
			// read in image data
			data[i].image.set_size(NUM_PXLS);
			for(int j = 0; j < NUM_PXLS; ++j)
			{
				data[i].image[j] = images.get(); 
			}
		}
	}

	srand(time(NULL)); 

}



// return random image data and labels...used for training neural network
std::vector<image_data> Data_manager::get_random_data(int num_data)
{
	using namespace std; 
	
	
	// vector to hold training data
	// random value to access from data reservoir and put into training data
	// random values used so as to not get duplicate data
	vector<image_data> sample(num_data); 
	int random_val = 0; 
	int* values_used = new int[num_data];
	
	// set values used to zero
	for(int i = 0; i < num_data; ++i)
		values_used[i] = 0; 
	
	// get random value, data at index of random value and put into 
	// training data vector. Make sure to not get the same random value twice. 
	for(int i = 0; i < num_data; ++i)
	{
		while(is_in(values_used, num_data, random_val))
			random_val = rand() % NUM_IMAGES; 

		sample[i] = data[random_val];

		values_used[i] = random_val; 
	}

	delete[] values_used; 
	
	return sample;
}


// helper function to detect if integer is in an array. 
bool is_in(int* array, int size, int number)
{
	for(int i = 0; i < size; ++i)
	{
		if(number == array[i])
			return true; 
	}
	return false; 
}

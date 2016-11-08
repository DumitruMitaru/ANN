#include "Data_manager.hpp"
#include <cstdlib>
#include <ctime>



// read and store training images and labels. 
void Data_manager::load()
{
	load_train_data(); 
	load_test_data(); 
	srand(time(NULL)); 

}


// load training data into train_data data member
void Data_manager::load_train_data()
{
	using namespace std; 

	ifstream train_images; // used to read training image data 
	ifstream train_labels; // used to read training label data for each image
	int label_val = 0;     // will hold current label value

	// open files and resize train_data to hold number of images to store
	train_images.open(TRAIN_IMAGE_FILE); 
	train_labels.open(TRAIN_LABEL_FILE); 
	train_data.resize(NUM_TRAIN_IMAGES); 

	if(train_images && train_labels)
	{
		// read past file headers
		for(int i = 0; i < 16; ++i)
			train_images.get(); 
		for(int i = 0; i < 8; ++i)
			train_labels.get(); 

		for(int i = 0; i < NUM_TRAIN_IMAGES; ++i)
		{
			// read in label data and set a vector of 10 elements to zero
			// setting the index value to the label value and setting that element 
			// at that index value to 1 
			train_data[i].label.set_size(10); 
			train_data[i].label.zeros();
			label_val = train_labels.get(); 
			train_data[i].label[label_val] = 1; 
			
			// read in image data
			train_data[i].image.set_size(NUM_PXLS);
			for(int j = 0; j < NUM_PXLS; ++j)
			{
				train_data[i].image[j] = train_images.get(); 
			}
		}


		train_images.close(); 
		train_labels.close(); 
	}

}



// load test data into test_data data member
void Data_manager::load_test_data()
{
	using namespace std;

	ifstream test_images;  // used to read test image data
	ifstream test_labels;  // used to read test label data for each image
	int label_val = 0;     // used to hold current label value

	// open files and resize test_data to hold number of images to store
	test_images.open(TEST_IMAGE_FILE); 
	test_labels.open(TEST_LABEL_FILE);
	test_data.resize(NUM_TEST_IMAGES); 

	if(test_images && test_labels)
	{
		// read past headers of files
		for(int i = 0; i < 16; ++i)
			test_images.get(); 
		for(int i = 0; i < 8; ++i)
			test_labels.get(); 
	
		// insert image and label data into test_data
		for(int i = 0; i < NUM_TEST_IMAGES; ++i)
		{
			// read in label data and set a vector of 10 elements to zero
			// setting the index value to the label value and setting that element 
			// at that index value to 1 
			test_data[i].label.set_size(10);
			test_data[i].label.zeros(); 
			label_val = test_labels.get();
			test_data[i].label[label_val] = 1; 

			test_data[i].image.set_size(NUM_PXLS);  // set size of array of image data

			for(int j = 0; j < NUM_PXLS; ++j)
			{
				test_data[i].image[j] = test_images.get(); // read in image data
			}
		}

		test_images.close(); 
		test_labels.close(); 
	}
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
			random_val = rand() % NUM_TRAIN_IMAGES; 

		sample[i] = train_data[random_val];
		values_used[i] = random_val; 
	}

	delete[] values_used; 
	
	return sample;
}



// return training_data paritioned into batches of size mini_batch_size
std::vector<std::vector<image_data> >  Data_manager::get_mini_batches(int mini_batch_size)
{
	int num_batches = train_data.size() / mini_batch_size; 
	int train_index = 0; 
	std::vector<image_data> batch(mini_batch_size); 		// mini-batch to hold subset of train_data
	std::vector<std::vector<image_data> > mini_batches(num_batches); // collection of mini batches 

	for(int i = 0; i < num_batches; ++i)
	{
		for(int j = 0; j < mini_batch_size; ++j)
		{
			batch[j] = train_data[train_index]; 
			++train_index; 
		}

		mini_batches[i] = batch; 
	}

	return mini_batches; 
}



// return test data
std::vector<image_data> Data_manager::get_test_data()
{
	return test_data; 
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
/*
using namespace std;

int ReverseInt (int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	
	ch1=i&255;
	ch2=(i>>8)&255;
	ch3=(i>>16)&255;
	ch4=(i>>24)&255;
	
	return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}



void ReadMNIST(int NumberOfImages, int DataOfAnImage,vector<vector<double>> &arr)
{
	arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
	ifstream image_file (IMAGE_FILE, ios::binary);

	if (image_file.is_open())
	{
		int magic_number=0;
		int number_of_images=0;
		int n_rows=0;
		int n_cols=0;
		
		image_file.read((char*)&magic_number,sizeof(magic_number));
		magic_number= ReverseInt(magic_number);
		image_file.read((char*)&number_of_images,sizeof(number_of_images));
		number_of_images= ReverseInt(number_of_images);
		image_file.read((char*)&n_rows,sizeof(n_rows));
		n_rows= ReverseInt(n_rows);
		image_file.read((char*)&n_cols,sizeof(n_cols));
		n_cols= ReverseInt(n_cols);
		
		for(int i = 0; i < number_of_images; ++i)
		{
			for(int r = 0; r < 784; ++r)
			{
					unsigned char temp=0;
					image_file.read((char*)&temp,sizeof(temp));
					arr[i][(n_rows*r)+c]= (double)temp;
			}
		}
	}
}

int main()
{
vector<vector<double>> ar;
ReadMNIST(10000,784,ar);

return 0;
}*/

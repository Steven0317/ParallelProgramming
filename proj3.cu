/*
*
*	Authors: Steven Faulkner, Blaine Oakley, Felipe Gutierrez
*
*	Final Project for CIS 4930, Implementation of K-means clustering
*	optimized with shared memory and reduction methods.
*
*	To compile nvcc kmeans.cu 
*	To run ./a.out "input.txt" "K" "iterations"
*
*	@data file: is the specified input file
*	@k: is the number of centroids tro be determined
*	@iterations: total number of iterations to be performed
*
*/

#include<cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

void ErrorCheck( cudaError_t err, const char op[]){
	/*
	*	@cudaError_t err: cuda api call catcher.
	*						* all cuda api's usually return 
	*						  cudaSuccess
	*	@const char op[]: error string will tell where api call
	*						failed
	*
	*	Error Catch Function, will wrap all malloc, memset and
	*		memcopy calls
	*	
	*/
		if( err != cudaSuccess ) 
			{ 
				printf("CUDA Error: %s, %s ", op, cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
	}



struct Point {
	/*
	*
	*	Struct for the imported data points
	*	
	*	Follows CADRe stanadards, moved cuadaMalloc and cudaFree 
	*       to the constructors, destrucor. Just makes the code look 
	*	a little cleaner
	*
	*
	*	@double x: x data point 
	*	@double y: y data point
	*	@int size: size ofdata point
	*	@int bytes: # of bytes allocated for storage
	*/
  
  Point(long dataSize) : dataSize(dataSize), num_of_bytes(dataSize * sizeof(double)) {
   
    ErrorCheck(cudaMalloc(&x, num_of_bytes),"Allocate x data\n");
    ErrorCheck(cudaMalloc(&y, num_of_bytes), "Allocate y data\n");
    ErrorCheck(cudaMemset(x, 0, num_of_bytes), "Set x data to '0'\n");
    ErrorCheck(cudaMemset(y, 0, num_of_bytes), "Set y data to '0'\n");

  }


  Point(long dataSize, std::vector<double>& x_data, std::vector<double>& y_data) : dataSize(dataSize), num_of_bytes(dataSize * sizeof(double)) {

    ErrorCheck(cudaMalloc(&x, num_of_bytes),"Allocate x array\n");
    ErrorCheck(cudaMalloc(&y, num_of_bytes), "Allocate y array\n");;
    ErrorCheck(cudaMemcpy(x, x_data.data(), num_of_bytes, cudaMemcpyHostToDevice),"Copy x array to device\n");
    ErrorCheck(cudaMemcpy(y, y_data.data(), num_of_bytes, cudaMemcpyHostToDevice), "Copy y array to device\n");

  }

  
  ~Point() {

    ErrorCheck(cudaFree(x),"Freeing x \n");
    ErrorCheck(cudaFree(y),"Freeing y \n");

  }
	
  double* x;
  double* y;
  long dataSize;
  int num_of_bytes;
};





__device__ double
euclidean_distance(double x_1, double y_1, double x_2, double y_2) {
/*
*
*	@double x_1, y_1, x_2, y_2: x and y coordinates from Point struct
*
*	
*	Standard Euclidean Distance function returns a straight line distance
*		Point A to Point B.
*	
*	//If I Have Time\\
*	We can exapnd this for higher dimensional data(add more x_n - x_m) or preprocess our data
*		with PCA(prinicpal component analysis) to reduce to 2 dimensions			
*
*/

  return sqrt((x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2));

}



__global__ void 
Assignment(double * data_x, double * data_y, int data_size, double * centroids_x, double * centroids_y, double * device_new_x, double * device_new_y, int k, int * device_counts) {
/*
*
*	@double* data_x: array of x data points
*	@double* data_y: array of y data points	
*	@int data_size: size of data array
*	@centroid_x: array of x centroids
*	@centroid_y: array of y centroids
*	@device_new_x: updated array for x
*	@device_new_y: updated array for y
*	@int k: # of centroids
*	@int* device_counts: int array, holds count 
*			for total points among all centodsi
*
*	K-Means Algorithm : each x,y cluster is assigned to its closest centroid
*						then each centroid is averaged over all the points
*						assigned to it and then updated with this new value
*/

  
  extern __shared__ double shared_mem[];

  int reg = threadIdx.x;

  int unique_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  //out of range
  if (unique_index >= data_size) return;

  //loading in centroids
  if (reg < k) {

  	//1D array seperated by K values for each point
    shared_mem[reg] = centroids_x[reg];
    shared_mem[k + reg] = centroids_y[reg];

  }

  __syncthreads();

  // load to registers 
  double x_value = data_x[unique_index];
  double y_value = data_y[unique_index];

  //none of our distance values will be large enough to use stl::infinity or FLT_MAX, 
  // arbitrary sentinal values will suffice for these two variables	
  double min_distance = 1000;
  int best_cluster = -1;
  
  //iterate over the all centroids keeping track of closest one for storage
  for (int cluster = 0; cluster < k; ++cluster) {
    
    double distance = euclidean_distance(x_value, y_value, shared_mem[cluster], shared_mem[k + cluster]);
   
    if (distance < min_distance) {

      min_distance = distance;

      best_cluster = cluster;

    }
  }

  __syncthreads();

  // tree-reduction start
  int x = reg;
  int y = reg + blockDim.x;
  int count = reg + blockDim.x + blockDim.x;

  //check if thread is assigned to centroid, writing to local memory if true or 0 if false
  for (int cluster = 0; cluster < k; ++cluster) {

    shared_mem[x] = (best_cluster == cluster) ? x_value : 0;

    shared_mem[y] = (best_cluster == cluster) ? y_value : 0;

    shared_mem[count] = (best_cluster == cluster) ? 1 : 0;
    
    __syncthreads();

    
    // Reduction for local memory.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {

      if (reg < stride) {

        shared_mem[x] += shared_mem[x + stride];

        shared_mem[y] += shared_mem[y + stride];

        shared_mem[count] += shared_mem[count + stride];

      }

      __syncthreads();

    }

    
    //push_back from shared mem to update array
    if (reg == 0) {

      int cluster_index = blockIdx.x * k + cluster;

      device_new_x[cluster_index] = shared_mem[x];
      device_new_y[cluster_index] = shared_mem[y];

      device_counts[cluster_index] = shared_mem[count];

    }
    __syncthreads();
  }
}






__global__ void 
centroid_recompute(double * centroids_x, double * centroids_y, double * device_new_x, double * device_new_y, int k,  int * device_counts) {
/*
*
*
*	@double * centroids_x: array for x centroids
*	@double * centroids_y: array for y centroids
*	@double * new_sums_x: updated x array
*	@double * new_sums_y: updated y array
*	@int k: # of centroids
*	@int * device_counts: int array,holds count 
*			for total points among all centodsi
*
*
*
*	centroid Recompute: Recomputes the centroids from all 
*						points assigned to it.
*
*
*/

  //local memory declaration
  extern __shared__ double shared_mem[];

  int reg = threadIdx.x;
  int b_Dim = blockDim.x;

  //load into local memory
  shared_mem[reg] = device_new_x[reg];
  shared_mem[b_Dim + reg] = device_new_y[reg];
  
  __syncthreads();

  //summination of every stride length block
  for (int stride = blockDim.x / 2; stride >= k; stride /= 2) {

    if (reg < stride) {

      shared_mem[reg] += shared_mem[reg + stride];
      shared_mem[b_Dim + reg] += shared_mem[b_Dim + reg + stride];

    }

    __syncthreads();

  }

  //recomputing centroid centers
  if (reg < k) {

    int count = max(1, device_counts[reg]);
   
    centroids_x[reg] = device_new_x[reg] / count;
    centroids_y[reg] = device_new_y[reg] / count;
    
    device_new_y[reg] = 0;
    device_new_x[reg] = 0;
    device_counts[reg] = 0;

  }
}






int main(int argc, const char * argv[]) {
  
  if (argc < 4) {
  
    std::cout << "Incorrect startup execution: <./a.out 'input.txt' 'K' 'iterations' " << std::endl;
    std::exit(EXIT_FAILURE);
  
  }

  int k = std::atoi(argv[2]);
  int number_of_iterations = std::atoi(argv[3]);

  std::vector<double> x_data;
  std::vector<double> y_data;
  std::ifstream stream_in(argv[1]);
  std::string line;
 
 if(stream_in){
  	
  	while (std::getline(stream_in, line)) {
    
    	std::istringstream line_stream(line);
    
    	double x, y;
    
    	line_stream >> x >> y;
    
    	x_data.push_back(x);
    	y_data.push_back(y);
  	}
 }
 else{

 	std::cout << "Error Opening File" << std::endl;
 	return(EXIT_FAILURE);
 }

  // dinput data up to 1,000,000 points
  long number_of_elements = x_data.size();


  // centroids are initalized to first k points of array
  // in order to chose 'randomly' we shuffle the array 
  // input array after we initilize the devize point array
  // and before we initilize the centroid array
  Point device_data(number_of_elements, x_data, y_data);

  std::srand(std::time(0));

  random_shuffle(x_data.begin(),x_data.end());
  random_shuffle(y_data.begin(),y_data.end());

  Point device_centroids(k, x_data, y_data);


  int threads = 1024;
  
  int blocks = (number_of_elements + threads - 1) / threads;


  std::cout << "\nProcessing " << number_of_elements << " points\n" << std::endl;

  
  int kmeans_shared_memory = 3 * threads * sizeof(double);
 
  int centroid_reduction_memory = 2 * k * blocks * sizeof(double);

 
  Point device_sum(k * blocks);
  
  int * device_count;
  
  ErrorCheck(cudaMalloc(&device_count, k * blocks * sizeof(int)), "Allocate size for device_count\n");
  ErrorCheck(cudaMemset(device_count, 0, k * blocks * sizeof(int)),"Set device_count to '0' \n");

  // cuda api time start
   cudaEvent_t start,stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start,0);



  // start iteration loop, assigning and updating centroid on each iteration
  for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
    

    Assignment<<<blocks, threads, kmeans_shared_memory>>>(device_data.x,device_data.y, device_data.dataSize, device_centroids.x, device_centroids.y, device_sum.x, device_sum.y, k,
                                                         																			device_count);
    cudaDeviceSynchronize();


    centroid_recompute<<<1, k * blocks, centroid_reduction_memory>>>(device_centroids.x, device_centroids.y, device_sum.x, device_sum.y, k, device_count);

    cudaDeviceSynchronize();
  }


  	// cuda api time stop
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	std::string unit = "";
	
	unit = (elapsedTime > 999) ? "seconds" : "milliseconds";

	elapsedTime = (elapsedTime > 999 ) ? elapsedTime/1000 : elapsedTime;

	std::cout << "Elapsed time of kernal calls: " << elapsedTime << " "  << unit << "\n" << std::endl;


  ErrorCheck(cudaFree(device_count),"Freeing Device Memory");
  
  std::vector<double> centroid_x(k, 0);
  std::vector<double> centroid_y(k, 0);

  ErrorCheck(cudaMemcpy(centroid_x.data(), device_centroids.x, device_centroids.num_of_bytes, cudaMemcpyDeviceToHost), "Moving Array back to host\n");
  ErrorCheck(cudaMemcpy(centroid_y.data(), device_centroids.y, device_centroids.num_of_bytes, cudaMemcpyDeviceToHost), "Moving Array back to host\n");

  
  std::cout << "centroids:" << std::endl;

  for (size_t cluster = 0; cluster < k; ++cluster) {
    std::cout << centroid_x[cluster] << " " << centroid_y[cluster] << std::endl;
  }
  std::cout << "\n" << std::endl;

  return(EXIT_SUCCESS);
}

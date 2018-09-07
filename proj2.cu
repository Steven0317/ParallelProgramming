/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the rc machines

	
   ==================================================================




	Steven Faulkner
	U9616-1844
	Summer 2018




*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram     */
long long	PDH_acnt;	/* total number of data points              */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of width                           */
atom * atom_list;		/* list of all data points                  */
int BlockSize;


struct timezone Idunno;
struct timeval startTime, endTime;



void ErrorCheck( cudaError_t err, const char op[])
	{
		if( err != cudaSuccess ) 
			{ 
				printf("CUDA Error: %s, %s ", op, cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
	}
/* 
	distance of two points in the atom_list 
*/

__device__ double 
p2p_distance(atom A, atom B)
{
	double x1 = A.x_pos;
	double x2 = B.x_pos;

	double y1 = A.y_pos;
	double y2 = B.y_pos;

	double z1 = A.z_pos;
	double z2 = B.z_pos;

	return  sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));

	
}
__global__ void
PDH_baseline(bucket *histo_in, atom *list, double width, int size)
{
	

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for(int x = i+1; x < size; ++x)
	{
		//double distance = p2p_distance(list,i,x);
//		int pos = (int) (distance/width);
		__syncthreads();
//		histo_in[pos].d_cnt++;
		__syncthreads();
	}
}
__global__ void
pdh_coal(bucket * histogram, atom * list, double width, int size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;




	while(i < size-1){
		
		for(int j = i+1; j < stride;++j){
//			double distance = p2p_distance(list,i,j);
//			int pos = (int) (distance/width);
//			atomicAdd(&histogram[pos].d_cnt,1);
			
		}
		i+=stride;
		
	}

}
__global__ void
pdh_priv(bucket * histogram,atom * list, double width, int size, int BlockSize)
{
	int t = threadIdx.x;
	int b = blockIdx.x;
	unsigned int reg = t + b * blockDim.x;
	extern __shared__ atom smem[];

	atom * private_atom = &smem[0];
	atom * localBlock = &smem[sizeof(private_atom)];	
	if(t < BlockSize)	
		private_atom[t] = list[reg];
		
	__syncthreads();
		
	//iterate over each block
	for(int i = b + 1; i < size/BlockSize; ++i)
	{
		
		unsigned int tempIdx = t + i * blockDim.x;
	
		if(tempIdx < BlockSize)	
			localBlock[t+ sizeof(private_atom)] = list[tempIdx];
				
		__syncthreads();
	

		//iterate through each thread within each block;
		for(int j =0; j < BlockSize;++j)
		{	
			double distance = p2p_distance(private_atom[t], localBlock[j]);
			int pos = (int) (distance/width);
			atomicAdd(&histogram[pos].d_cnt,1);
		}
		
	}
	
	for(int i = t+1;i < BlockSize;++i)
	{
		double distance = p2p_distance(private_atom[t],private_atom[i]);
		int pos = (int) (distance/width);
		atomicAdd( &histogram[pos].d_cnt,1);
	} 

}
void output_histogram(bucket *histogram){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
		
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("|\n T: %lld \n ", total_cnt);
		else printf("| ");
	}
}

int main(int argc, char **argv)
{
	int i;

	if(argc == 4){

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	BlockSize = atoi(argv[3]);
	printf("Parameters are %d, %.2f, %d", PDH_acnt,PDH_res,BlockSize);
	
	}
	
	else{

	int count = argc -1;		
	printf("Too Few Arguments to Function, required 3 only recieved %d\n\n", count);
	exit(EXIT_FAILURE);
	}

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	size_t hist_size = sizeof(bucket)*num_buckets;
	size_t atom_size = sizeof(atom)*PDH_acnt;



	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	
	
	bucket *dev_Histo = NULL;
	atom *dev_atomL = NULL;

	ErrorCheck(cudaMalloc((void**) &dev_Histo,hist_size), "Allocate Memory for Histogram");
	ErrorCheck(cudaMalloc((void**) &dev_atomL, atom_size), "Allocate Memory for Atom List");
	ErrorCheck(cudaMemcpy(dev_Histo,histogram,hist_size, cudaMemcpyHostToDevice), "Copying Histogram to Device");
	ErrorCheck(cudaMemcpy(dev_atomL, atom_list, atom_size, cudaMemcpyHostToDevice), "Copying Atom list to Device");
	
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);


//	PDH_baseline <<<ceil(PDH_acnt/32), 32 >>> (dev_Histo, dev_atomL, PDH_res, PDH_acnt);

	pdh_priv <<<ceil(PDH_acnt/BlockSize),BlockSize,2 *  BlockSize*sizeof(atom) >>>(dev_Histo, dev_atomL,PDH_res,PDH_acnt,BlockSize);
	

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	ErrorCheck(cudaMemcpy(histogram, dev_Histo, hist_size, cudaMemcpyDeviceToHost), " Move Histogram back to Host");
		
	/* print out the histogram */
	output_histogram(histogram);

	//elapsedTime = elapsedTime / 1000;
	printf("\n\n******* Total Running Time of Kernal = %.5f milliseconds *******\n\n", elapsedTime);
	

	ErrorCheck(cudaFree(dev_Histo), "Free Device Histogram");
	ErrorCheck(cudaFree(dev_atomL), "Free Device Atom List");


	ErrorCheck(cudaDeviceReset(), "Reset");
	
	free(histogram);
	free(atom_list);
	
	return 0;
}



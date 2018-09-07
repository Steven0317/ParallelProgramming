/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the rc machines

	StevenFaulkner U9616-1844
	Summer 2018
   ==================================================================
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
p2p_distance(atom *list, int ind1, int ind2)
{
	double x1 = list[ind1].x_pos;
	double x2 = list[ind2].x_pos;

	double y1 = list[ind1].y_pos;
	double y2 = list[ind2].y_pos;

	double z1 = list[ind1].z_pos;
	double z2 = list[ind2].z_pos;

	return  sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));

	
}
/* 
	brute-force SDH solution in a single CPU thread 
*/
__global__ void
PDH_baseline(bucket *histo_in, atom *list, double width, int size)
{
	int i, j, pos;
	double distance;

	i = (blockIdx.x * blockDim.x) + threadIdx.x;
	j = i+1;
	
	for(int x = j; x < size; ++x)
	{
		distance = p2p_distance(list,i,x);
		pos = (int) (distance/width);
		atomicAdd( &histo_in[pos].d_cnt,1);
	}
}

__global__ void 
PDHGPU_Baseline(bucket *histogram,atom *list, double width)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if(x < y)
	 {
		double dist = p2p_distance(list,x,y);
		int pos = (int) (dist/width);
		histogram[pos].d_cnt++;
		printf("%d,%d : %d, %f \n", x,y,pos,dist);
	 }

	__syncthreads();

}
/* 
	set a checkpoint and show the (natural) running time in seconds 
*/

double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/
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
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);


	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	size_t  hist_size = sizeof(bucket)*num_buckets;
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
	/*	
	PDH_baseline();
	report_running_time();
	output_histogram(histogram);
	*/
	
	bucket *dev_Histo = NULL;
	atom *dev_atomL = NULL;

	ErrorCheck(cudaMalloc((void**) &dev_Histo,hist_size), "Allocate Memory for Histogram");
	ErrorCheck(cudaMalloc((void**) &dev_atomL, atom_size), "Allocate Memory for Atom List");
	ErrorCheck(cudaMemcpy(dev_Histo,histogram,hist_size, cudaMemcpyHostToDevice), "Copying Histogram to Device");
	ErrorCheck(cudaMemcpy(dev_atomL, atom_list, atom_size, cudaMemcpyHostToDevice), "Copying Atom list to Device");

	

	PDH_baseline <<<ceil(PDH_acnt/32), 32 >>> (dev_Histo, dev_atomL, PDH_res, PDH_acnt);


	ErrorCheck(cudaMemcpy(histogram, dev_Histo, hist_size, cudaMemcpyDeviceToHost), " Move Histogram to host");
		
	/* print out the histogram */
	output_histogram(histogram);
	

	ErrorCheck(cudaFree(dev_Histo), "Free Device Histogram");
	ErrorCheck(cudaFree(dev_atomL), "Free Device Atom List");

	free(histogram);
	free(atom_list);

	ErrorCheck(cudaDeviceReset(), "Reset");
	return 0;
}



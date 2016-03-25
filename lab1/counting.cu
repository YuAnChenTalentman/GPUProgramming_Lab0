#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }


__device__ int lowbit(int x)
{
	return x&(-x);
}

__global__ void BIT(const char *text, int *pos, int text_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i=idx,done=1; i<text_size && done==0; i+=512)
    {
        for (int j=i, done=0; j>(i-lowbit(i)) && done==0; j--)
        {
        	char ch=text[j];
        	if(ch != 32)
        	{
        		pos[j]+=1;
        	}
        	else
        	{
        		done=1;
        	}
        }
    }
}


struct head_filter : public thrust::unary_function<thrust::tuple<int, int>, int>
{
  __host__ __device__
  int operator()(const thrust::tuple<int, int>& x) const
  {
		if (x.get<0>() == 1)
			return x.get<1>();

		else return -1;
   }
};

 
void CountPosition(const char *text, int *pos, int text_size)
{
	BIT<<<16,32>>>(text, pos, text_size);
}

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);
	thrust::device_vector<int> pos_vector(pos_d, pos_d+text_size);
	thrust::device_vector<int> head_vector(head_d, head_d+text_size);
	// TODO
	thrust::counting_iterator<int> idxfirst(0);
  	thrust::counting_iterator<int> idxlast = idxfirst + text_size;
  	head_filter my_head_filter;
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(pos_vector.begin(), idxfirst)), 
    	thrust::make_zip_iterator(thrust::make_tuple(pos_vector.end(), idxlast)), head_vector.begin(), my_head_filter);
	head_vector.erase(remove(head_vector.begin(), head_vector.end(), -1), head_vector.end());
	nhead = head_vector.size();
	thrust::device_vector<int> output_vector(nhead);
	thrust::copy(head_vector.begin(), head_vector.end(), output_vector.begin());
	head = thrust::raw_pointer_cast(output_vector.data());
	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}

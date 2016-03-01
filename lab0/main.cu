#include <cstdio>
#include <cstdlib>
#include "SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

__global__ void Capitalize(char *input_gpu, int fsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	while(idx<fsize)
	{		
		char ch = input_gpu[idx];
		input_gpu[idx] = (ch>96&&ch<123)?(ch-32):ch;		
		idx+=512;


	}
}
__global__ void Swap_Pair(char *input_gpu, int fsize) {
	int idx = (blockIdx.x * blockDim.x + threadIdx.x)*2;
	while(idx<fsize)
	{		
		char aux;


		aux = input_gpu[idx];
		input_gpu[idx] = input_gpu[idx+1];
		input_gpu[idx+1] = aux;

		//input_gpu[idx] = (ch>96&&ch<123)?(ch-32):ch;		
		idx+=1024;


	}
}



int main(int argc, char **argv)
{
	// init, and check
	if (argc != 2) {
		printf("Usage %s <input text file>\n", argv[0]);
		abort();
	}
	FILE *fp = fopen(argv[1], "r");
	if (!fp) {
		printf("Cannot open %s", argv[1]);
		abort();
	}
	// get file size
	fseek(fp, 0, SEEK_END);
	size_t fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	// read files
	MemoryBuffer<char> text(fsize+1);
	auto text_smem = text.CreateSync(fsize);
	CHECK;
	fread(text_smem.get_cpu_wo(), 1, fsize, fp);
	text_smem.get_cpu_wo()[fsize] = '\0';
	fclose(fp);

	// TODO: do your transform here
	char *input_gpu = text_smem.get_gpu_rw();
	// An example: transform the first 64 characters to '!'
	// Don't transform over the tail
	// And don't transform the line breaks
	//Capitalize<<<16,32>>>(input_gpu, fsize);
	Swap_Pair<<<16,32>>>(input_gpu,fsize);
	puts(text_smem.get_cpu_ro());
	int d =0;
	scanf("%d",d);
	return 0;
}

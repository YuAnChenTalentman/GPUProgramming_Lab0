#include "lab3.h"
#include <cstdio>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SyncedMemory.h"

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void Solver(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox,
	int *R, float *b, int *D)
{
	int Np = 0;
	int N = wt*ht;
	int dP[4][2] = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy + yt, xb = ox + xt;
		const int curb = wb*yb + xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			int tValueR = target[curt * 3 + 0];
			int tValueG = target[curt * 3 + 1];
			int tValueB = target[curt * 3 + 2];
			for (int k = 0; k < 3; k++)
			{
				b[curt * 3 + k] = 0;
			}
			for (int k = 0; k < 4; k++)
			{
				int xt2 = xt + dP[k][0];
				int yt2 = yt + dP[k][1];
				int curt2 = wt*yt2 + xt2;
				int yb2 = oy + yt2, xb2 = ox + xt2;
				int curb2 = wb*yb2 + xb2;
				R[curt*4 + k] = -1;
				Np++;
				if (xt2<0|| yt2<0 ||mask[curt2] < 127.0f)
				{
						
					b[curt * 3 + 0] += output[curb2 * 3 + 0];
					b[curt * 3 + 1] += output[curb2 * 3 + 1];
					b[curt * 3 + 2] += output[curb2 * 3 + 2];
				}
				else
				{
						
					R[curt * 4 + k] = curb2;
					int qValueR = target[curt2 * 3 + 0];
					int qValueG = target[curt2 * 3 + 1];
					int qValueB = target[curt2 * 3 + 2];
					b[curt * 3 + 0] += tValueR - qValueR;
					b[curt * 3 + 1] += tValueG - qValueG;
					b[curt * 3 + 2] += tValueB - qValueB;
				}
			}
			D[curt] = Np;
		}
	}
}


__global__ void JacobiIteration(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox,
	int *R, float *b, int *D)
{
	int N = wt*ht;
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;
	
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy + yt, xb = ox + xt;
		const int curb = wb*yb + xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			float nextR,nextG,nextB;
			nextR = b[curt * 3 + 0];
			nextG = b[curt * 3 + 1];
			nextB = b[curt * 3 + 2];
			for (int n = 0; n < 4; n++) {
				int index = R[curt * 4 + n];
				if (index >= 0)
				{
					nextR += output[index * 3 + 0];
					nextG += output[index * 3 + 1];
					nextB += output[index * 3 + 2];
				}
			}
			nextR /= (D[curt]);
			nextG /= (D[curt]);
			nextB /= (D[curt]);
			output[curb*3 + 0] = nextR;
			output[curb*3 + 1] = nextG;
			output[curb*3 + 2] = nextB;
		}
	}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	int N = wt*ht;
	MemoryBuffer<int>  myR(N * 4),myD(N);
	MemoryBuffer<float> myb(N * 3);
	MemoryBuffer <float>CloneOutput(wb*hb * 3);
	auto R_s = myR.CreateSync(N * 4);
	auto D_s = myD.CreateSync(N);
	auto b_s = myb.CreateSync(N * 3);
	auto CloneOutput_s = CloneOutput.CreateSync(wb*hb * 3);
	float *CloneOutput_gpu = CloneOutput_s.get_gpu_rw();
	int *R_gpu = R_s.get_gpu_rw();
	int *D_gpu = D_s.get_gpu_rw();
	float *b_gpu = b_s.get_gpu_rw();
	cudaMemcpy(CloneOutput_gpu, background, wb*hb*sizeof(float) * 3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, CloneOutput_gpu,
		wb, hb, wt, ht, oy, ox
	);
	Solver << <dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16) >> >(
		background, target, mask, CloneOutput_gpu,
		wb, hb, wt, ht, oy, ox, R_gpu, b_gpu, D_gpu
		);
	for (int i = 0; i < 20000; i++)
	{
		JacobiIteration << <dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16) >> >(
			background, target, mask, CloneOutput_gpu,
			wb, hb, wt, ht, oy, ox, R_gpu, b_gpu, D_gpu
		);
	}
	cudaMemcpy(output, CloneOutput_gpu, wb*hb*sizeof(float) * 3, cudaMemcpyDeviceToDevice);
	cudaFree(R_gpu);
	cudaFree(D_gpu);
	cudaFree(b_gpu);
	cudaFree(CloneOutput_gpu);
	
}

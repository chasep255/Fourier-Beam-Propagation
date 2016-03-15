#include <thrust/complex.h>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <cufft.h>
#include <sys/time.h>
#include "FourierBeamPropagator.cuh"

typedef double real;
typedef thrust::complex<real> complex;

const int N = 2048;
const int ITS = 10000;
const double lambda = 500.0e-9;
const double dz = 1.0e-6;
const double n = 1.0; 
const double k0 = 2.0 * M_PI / lambda;
const double grid_size = 0.25 * lambda * N;
const double over_grid_size = 1.0 / grid_size;
const double over_N2 = 1.0 / (N * N);

real white_value = 0.25;

void save(const char* fn, complex* g)
{	
	unsigned char* buffer = new unsigned char[N * N];
	
//	for(int i = 0; i < N * N; i++)
//		white_value = fmax(white_value, thrust::abs(g[i]));
//	std::cout << white_value << std::endl;
	real over_white = 1.0 / white_value;
	
	
	for(int i = 0; i < N * N; i++)
		buffer[i] = fmin(255.0, 255.0 * thrust::abs(g[i]) * over_white);
	
	std::string cmd = std::string("convert /dev/stdin ") + fn;
	FILE* out = popen(cmd.c_str(), "w");
	
	fprintf(out, "P5 %d %d 255 ", N, N);
	
	fwrite(buffer, 1, N * N, out);
	
	delete[] buffer;
	fclose(out);
}

inline void cucheck()
{
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(err));
	}
}

void initE(complex* efld)
{
	for(int y = 0; y < N; y++)
	{
		for(int x = 0; x < N; x++)
		{
			int dx = x - N / 2;
			int dy = y - N / 2;
//			
//			if(dx * dx + dy * dy < 100 * 100)
//				efld[y * N + x] = 1.0;
//			else
//				efld[y * N + x] = 0.0;
//			
//			efld[y * N + x] = exp(-(dx * dx + dy * dy) / 60.0);
			
			if((x == 900 || x == 1100) && y >= 900 && y < 1100)
				efld[y * N + x] = 1.0;
			else
				efld[y * N + x] = 0.0;
		}
	}
}

__global__ void refraction_kernel(complex* efld)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if(idx < N && idy < N)
	{
		efld[idy * N + idx] *= thrust::exp(complex(0.0, -n * dz * k0));
	}
}

__global__ void diffraction_kernel(complex* efld_fft)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if(idx < N && idy < N)
	{
		real kx = idx > N / 2 ? (idx - N) * over_grid_size : idx * over_grid_size;
		real ky = idy > N / 2 ? (idy - N) * over_grid_size : idy * over_grid_size;
		real kz = sqrt(k0 * k0 - kx * kx - ky * ky);
		real phi = -dz * kz;
		real re, im;
		sincos(phi, &im, &re);
		efld_fft[idy * N + idx] *= complex(re * over_N2, im * over_N2);
	}
}

double prof_time()
{
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_sec + 1.0e-6 * now.tv_usec;
}

int main()
{
	
	cudaSetDevice(0);
	complex* elec_fld = new complex[N * N];
	initE(elec_fld);
	
	complex* elec_fld_d;
	complex* elec_fft_d;
	cudaMalloc(&elec_fld_d, sizeof(complex) * N * N);
	cucheck();
	
	cudaMalloc(&elec_fft_d, sizeof(complex) * N * N);
	cucheck();
	
	cufftHandle plan;
	if(cufftPlan2d(&plan, N, N, CUFFT_Z2Z) != CUFFT_SUCCESS)
		exit(1);
	
	cudaMemcpy(elec_fld_d, elec_fld, sizeof(complex) * N * N, cudaMemcpyHostToDevice);
	cucheck();
	
	int refraction_threads;
	int refraction_blocks;
	int diffraction_threads;
	int diffraction_blocks;
	
	cudaFuncAttributes attrs;
	cudaFuncGetAttributes(&attrs, diffraction_kernel);
	cucheck();
	refraction_threads = sqrt(attrs.maxThreadsPerBlock);
	refraction_blocks = (N + (refraction_threads - 1)) / refraction_threads;
	
	cudaFuncGetAttributes(&attrs, diffraction_kernel);
	cucheck();
	diffraction_threads = sqrt(attrs.maxThreadsPerBlock);
	diffraction_blocks = (N + (diffraction_threads - 1)) / diffraction_threads;
	
	std::cout << diffraction_threads << "\t" << diffraction_blocks << std::endl;
	std::cout << diffraction_threads << "\t" << diffraction_blocks << std::endl;
	
	save("pattern.jpg", elec_fld);
	
	std::cout << sizeof(cufftDoubleComplex) << "\t" << sizeof(complex) << std::endl;
	int file_number = 0;
	double start = prof_time();
	for(int i = 0; i < ITS; i++)
	{
		std::cout << i << std::endl;
		refraction_kernel<<< dim3(refraction_blocks, refraction_blocks), dim3(refraction_threads, refraction_threads) >>>(elec_fld_d);
		cucheck();
		if(cufftExecZ2Z(plan, (cufftDoubleComplex*)elec_fld_d, (cufftDoubleComplex*)elec_fft_d, CUFFT_FORWARD) != CUFFT_SUCCESS)
			exit(1);
		diffraction_kernel<<< dim3(diffraction_blocks, diffraction_blocks), dim3(diffraction_threads, diffraction_threads) >>>(elec_fft_d);
		cucheck();
		if(cufftExecZ2Z(plan, (cufftDoubleComplex*)elec_fft_d, (cufftDoubleComplex*)elec_fld_d, CUFFT_INVERSE) != CUFFT_SUCCESS)
			exit(1);
		
		if(i % 4 == 0)
		{
			cucheck();
			char file_name[100];
			sprintf(file_name, "out/img%d.jpg\n", file_number++);
			save(file_name, elec_fld);
			cudaMemcpy(elec_fld, elec_fld_d, sizeof(complex) * N * N, cudaMemcpyDeviceToHost);
		}
			
	}
	double end = prof_time();
	std::cout << (end - start) << std::endl;
	cufftDestroy(plan);
	cudaFree(elec_fft_d);
	cudaFree(elec_fft_d);
	delete[] elec_fld;
	return 0;
}

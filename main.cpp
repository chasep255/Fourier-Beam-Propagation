#include <complex>
#include <cmath>
#include <iostream>
#include <stddef.h>
#include "fourier.hpp"
#include <omp.h>
#include <fstream>
#include <fftw3.h>
#include <cstring>
#include <limits>
#include <string>
#include <cassert>
#include <thread>

const int N = 512;
static std::thread last_thread;
double white_val;

void save(const char* fn, std::complex<double>* g)
{	
	if(last_thread.joinable())
		last_thread.join();
	
	white_val = 0.0;
	#pragma omp parallel for reduction(max:white_val)
	for(int i = 0; i < N * N; i++)
		white_val = fmax(white_val, (g[i] * std::conj(g[i])).real());
	
	std::cout << white_val << std::endl;
	
	unsigned char* buffer = new unsigned char[N * N];
	double over_white = 1.0 / white_val;
	#pragma omp parallel for
	for(int i = 0; i < N * N; i++)
		buffer[i] = fmin(255.0, 255.0 * (g[i] * std::conj(g[i])).real() * over_white);
	
	std::string fncp = fn;
	last_thread = std::thread([buffer, fncp]()
	{
		std::string cmd = "convert /dev/stdin " + fncp;
		FILE* out = popen(cmd.c_str(), "w");
		assert(out);
		
		fprintf(out, "P5 %d %d 255 ", N, N);
		
		fwrite(buffer, 1, N * N, out);
		
		delete[] buffer;
		fclose(out);
	});
}

int main()
{	
	fftw_init_threads();
	const int THREADS = 4;
	fftw_plan_with_nthreads(THREADS);
	omp_set_num_threads(THREADS);
	omp_set_dynamic(true);
	
	white_val = 0.8;
	
	typedef std::complex<double> complex;
	const complex I(0.0, 1.0);
	
	complex* e_prime = new std::complex<double>[N * N];
	complex* e = new std::complex<double>[N * N];
	
	memset(e, 0, sizeof(std::complex<double>) * N * N);
	
	const double lambda = 500.0e-9;
	const double dz = 1.0;
	const double n = 1.0; 
	const double k0 = 2.0 * M_PI / lambda;
	const double grid_size = 0.25 * lambda * N;
	const double over_grid_size = 1.0 / grid_size;
	const double over_N2 = 1.0 / (N * N);
	
	const int ITS = 300000;
	
	fftw_plan forwards = fftw_plan_dft_2d(N, N, (fftw_complex*)e, (fftw_complex*)e_prime, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan backwards = fftw_plan_dft_2d(N, N, (fftw_complex*)e_prime, (fftw_complex*)e, FFTW_BACKWARD, FFTW_MEASURE);
	
	#pragma omp parallel for
	for(int y = 0; y < N; y++)
	{
		for(int x = 0; x < N; x++)
		{
			int dx = x - N / 2;
			int dy = y - N / 2;
			
			e[y * N + x]=std::exp(-(dx * dx + dy * dy) / 60.0);
		}
	}
	
	int saven = 0;
	double start = omp_get_wtime();
	for(int i = 0; i < ITS; i++)
	{
		std::cout << "Iteration " << (i + 1) << " / " << ITS  << std::endl;
		complex r = std::exp(-I * n * dz * k0);
		#pragma omp parallel for
		for(int y = 0; y < N; y++)
		{
			for(int x = 0; x < N; x++)
			{
				int dx = x - N / 2;
				int dy = y - N / 2;
				
				if(dx * dx + dy * dy > 200 * 200)
				{
					e[y * N + x] = 0;
				}
				else
					e[y * N + x] *= r;
			}
		}
		
		fftw_execute(forwards);

		#pragma omp parallel for
		for(int y = 0; y < N; y++)
		{
			for(int x = 0; x < N; x++)
			{
				double kx = (x - N / 2) * over_grid_size;
				double ky = (y - N / 2) * over_grid_size;
				double kz = std::sqrt(k0 * k0 - kx * kx - ky * ky);
				
				double phi = -dz * kz;
				complex ex(cos(phi), sin(phi));
				e_prime[y * N + x] *= ex * over_N2;
			}
		}
		
		fftw_execute(backwards);
		
		if(i % 1 == 0)
		{
			char str[100];
			sprintf(str, "out/img%d.jpg\n", saven++);
			save(str, e);
		}
	}
	double end = omp_get_wtime();
	
	std::cout << "dt = " << (end - start) << std::endl;
	
	delete[] e;
	delete[] e_prime;
	
	fftw_destroy_plan(forwards);
	fftw_destroy_plan(backwards);
	fftw_cleanup_threads();
	
	last_thread.join();
	return 0;
}

#ifndef _FOURIER_BEAM_PROPAGATOR_
#define _FOURIER_BEAM_PROPAGATOR_

#include <cufft.h>
#include <thrust/complex.h>
#include <stdexcept>
#include <algorithm>
#include <type_traits>

namespace fftbpm
{	
	template<typename real, typename IndexOfRefractionFunct>
	__global__ void _step_size_kernel(IndexOfRefractionFunct n, real dz, unsigned int* sub_steps, 
			double z, real delta_threshold, real cell_size, int cells)
	{
		typedef thrust::complex<real> complex;
		__shared__ unsigned int block_max;
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= cells * cells)
			return;
		
		int idy = idx / cells;
		idx %= cells;
		
		real x = idx * cell_size - 0.5f * cell_size * cells;
		real y = idy * cell_size - 0.5f * cell_size * cells;
		
		real dn = thrust::abs(n(x, y, z) - n(x, y, z + dz));
		int steps = max(1.0, ceil(dn / delta_threshold));
		
		if(threadIdx.x == 0)
			block_max = steps;
		
		__syncthreads();
		if(steps > block_max)
			atomicMax(&block_max, steps);
		
		__syncthreads();
		if(threadIdx.x == 0)
			atomicMax(sub_steps, block_max);
	}
	
	template<typename real, typename IndexOfRefractionFunct>
	__global__ void _refraction_kernel(thrust::complex<real>* efld, IndexOfRefractionFunct n, 
			real dz, double z, real cell_size, real k0, int border_size, int cells)
	{
		typedef thrust::complex<real> complex;
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= cells * cells)
			return;
		
		int idy = idx / cells;
		idx %= cells;
		
		real x = cell_size * idx - 0.5f * cell_size * cells;
		real y = cell_size * idy - 0.5f * cell_size * cells;
		
		complex n_avg = complex(n(x, y, z)) / (real)3.0f;
		
		const complex I(0, 1);
		
		complex ef = efld[idy * cells + idx];
		ef *= thrust::exp(-I * n_avg * dz * k0);
		
		int r = min(idx, min(idy, min(cells - idx, cells - idy)));
		if(r < border_size)
		{
			float dem = 1.0f / (0.66f * border_size * border_size);
			r = border_size - r;
			ef *= expf(-r * r * dem);
		}
		
		efld[idy * cells + idx] = ef;
	}
	
	template<typename real>
	__global__ void _diffraction_kernel(thrust::complex<real>* efld_fft, real dz, real cell_size, 
			real k0, int cells)
	{
		typedef thrust::complex<real> complex;
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if(idx >= cells * cells)
			return;
		
		int idy = idx / cells;
		idx %= cells;
		
		real kx = (real)2.0 * (real)M_PI * (idx > cells / 2 ? (idx - cells) : idx) / (cell_size * cells);
		real ky = (real)2.0 * (real)M_PI * (idy > cells / 2 ? (idy - cells) : idy) / (cell_size * cells);
		real kz = sqrt(k0 * k0 - kx * kx - ky * ky);
		
		real re, im;
		sincos(-dz * kz, &im, &re);
		efld_fft[idy * cells + idx] *= complex(re, im) / (real)(cells * cells);
	}
	
	template<typename real, typename IndexOfRefractionFunct>
	class BeamPropagator
	{
		public:
		typedef thrust::complex<real> complex;
		
		BeamPropagator(int _cells, real lambda, real _cell_size = 1.0, 
				real _z = 0.0, IndexOfRefractionFunct _n = IndexOfRefractionFunct()) :
			cells(_cells), n(_n), cell_size(_cell_size), z(_z), k0(2.0 * M_PI / lambda)
		{
			cudaGetDevice(&device);
			cucheck();
			
			cufftcheck(cufftPlan2d(&plan, cells, cells, std::is_same<double, real>::value ? CUFFT_Z2Z : CUFFT_C2C));
			
			size_t cells2 = (size_t)_cells * _cells;
			
			cudaMalloc(&fft_d, sizeof(complex) * cells2);
			cucheck();
			
			cudaMalloc(&efld_d, sizeof(complex) * cells2);
			cucheck();
			
			cudaMalloc(&n_buffer_d, sizeof(complex) * cells2);
			cucheck();
			
			cudaMalloc(&sub_steps_d, sizeof(unsigned int));
			cucheck();
			
			cudaFuncAttributes sattrs, dattrs, rattrs;
			cudaFuncGetAttributes(&sattrs, _step_size_kernel<double, IndexOfRefractionFunct>);
			cucheck();
			cudaFuncGetAttributes(&rattrs, _refraction_kernel<double, IndexOfRefractionFunct>);
			cucheck();
			cudaFuncGetAttributes(&dattrs, _diffraction_kernel<double>);
			cucheck();
			
			step_threads= sattrs.maxThreadsPerBlock;
			step_blocks = (cells2 + (step_threads - 1)) / step_threads;
			refraction_threads= rattrs.maxThreadsPerBlock;
			refraction_blocks = (cells2 + (refraction_threads - 1)) / refraction_threads;
			diffraction_threads= dattrs.maxThreadsPerBlock;
			diffraction_blocks = (cells2 + (diffraction_threads - 1)) / diffraction_threads;
		}
		
		~BeamPropagator()
		{
			cufftDestroy(plan);
			cudaFree(fft_d);
			cudaFree(efld_d);
			cudaFree(n_buffer_d);
			cudaFree(sub_steps_d);
			
			plan = 0;
			fft_d = nullptr;
			efld_d = nullptr;
			n_buffer_d = nullptr;
			sub_steps_d = nullptr;
		}
		
		void setElectricField(const complex* e)
		{
			cudaSetDevice(device);
			cucheck();
			cudaMemcpy(efld_d, e, sizeof(complex) * cells * cells, cudaMemcpyHostToDevice);
			cucheck();
		}
		
		void getElectricField(complex* e)
		{
			cudaSetDevice(device);
			cucheck();
			cudaMemcpy(e, efld_d, sizeof(complex) * cells * cells, cudaMemcpyDeviceToHost);
			cucheck();
		}
		
		complex* getElectricFieldDevice() const
		{
			return efld_d;
		}
		
		int getGridDim() const
		{
			return cells;
		}
		
		int getBorderSize() const
		{
			return border;
		}
		
		void setBorderSize(unsigned int size) const
		{
			border = size;
		}
		
		real getPhysicalGridDim() const
		{
			return cells * cell_size;
		}
		
		real getCellSize() const
		{
			return cell_size;
		}
		
		real getWaveVectorMag() const
		{
			return k0;
		}
		
		int getDevice() const
		{
			return device;
		}
		
		double getZCoordinate()const
		{
			return z;
		}
		
		complex indexOfRefraction(real x, real y, real z) const
		{
			return n(x, y, z);
		}
		
		unsigned long long getStepCount() const
		{
			return steps;
		}
		
		real getMaxNChange() const
		{
			return max_n_change;
		}
		
		void setMaxNChange(real n)
		{
			max_n_change = n;
		}
		
		void step(real dz, bool delta_check = true, cudaStream_t stream = 0)
		{
			cudaSetDevice(device);
			cucheck();
			cufftcheck(cufftSetStream(plan, stream));
			
			unsigned int sub_steps = 1;
			if(delta_check)
			{
				cudaMemsetAsync(sub_steps_d, 0, sizeof(unsigned int), stream);
				cucheck();
				_step_size_kernel<real, IndexOfRefractionFunct><<< step_blocks, step_threads, 0, stream>>>(n, dz, sub_steps_d, z, max_n_change, cell_size, cells);
				cucheck();
				cudaMemcpy(&sub_steps, sub_steps_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
				cucheck();
			}
			
			dz /= sub_steps;
			for(int i = 0; i < sub_steps; i++)
			{
				_refraction_kernel<real, IndexOfRefractionFunct><<< refraction_blocks, refraction_threads, 0, stream >>>(efld_d, n, dz, z, cell_size, k0, border, cells);
				cucheck();
				
				if(std::is_same<double, real>::value)
					cufftcheck(cufftExecZ2Z(plan, (cufftDoubleComplex*)efld_d, (cufftDoubleComplex*)fft_d, CUFFT_FORWARD));
				else
					cufftcheck(cufftExecC2C(plan, (cufftComplex*)efld_d, (cufftComplex*)fft_d, CUFFT_FORWARD));
				
				_diffraction_kernel<real><<< diffraction_blocks, diffraction_threads, 0, stream >>>(fft_d, dz, cell_size, k0, cells);
				cucheck();
				
				if(std::is_same<double, real>::value)
					cufftcheck(cufftExecZ2Z(plan, (cufftDoubleComplex*)fft_d, (cufftDoubleComplex*)efld_d, CUFFT_INVERSE));
				else
					cufftcheck(cufftExecC2C(plan, (cufftComplex*)fft_d, (cufftComplex*)efld_d, CUFFT_INVERSE));
				
				z += dz;
			}
			cudaStreamSynchronize(stream);
			cucheck();
			steps++;
		}
		
		private:
		int device;
		int cells;
		int border = 10;
		IndexOfRefractionFunct n;
		cufftHandle plan = 0;
		real cell_size;
		double z;
		real k0;
		real max_n_change;
		unsigned long long steps = 0;
		complex* fft_d = nullptr;
		complex* efld_d = nullptr;
		complex* n_buffer_d = nullptr;
		unsigned int* sub_steps_d = nullptr;
		
		int step_blocks;
		int step_threads;
		int refraction_blocks;
		int refraction_threads;
		int diffraction_blocks;
		int diffraction_threads;
		
		static void cucheck()
		{
			cudaError_t err = cudaGetLastError();
			if(err != cudaSuccess)
			{
				throw std::runtime_error(cudaGetErrorString(err));
			}
		}
		
		static void cufftcheck(cufftResult r)
		{
			if(r == CUFFT_SUCCESS)
				return;
			
			switch(r)
			{
				case CUFFT_INVALID_PLAN:
					throw std::runtime_error("CUFFT_INVALID_PLAN");
				case CUFFT_ALLOC_FAILED:
					throw std::runtime_error("CUFFT_ALLOC_FAILED");
				case CUFFT_INVALID_TYPE:
					throw std::runtime_error("CUFFT_INVALID_TYPE");
				case CUFFT_INVALID_VALUE:
					throw std::runtime_error("CUFFT_INVALID_VALUE");
				case CUFFT_INTERNAL_ERROR:
					throw std::runtime_error("CUFFT_INTERNAL_ERROR");
				case CUFFT_EXEC_FAILED:
					throw std::runtime_error("CUFFT_EXEC_FAILED");
				case CUFFT_SETUP_FAILED:
					throw std::runtime_error("CUFFT_SETUP_FAILED");
				case CUFFT_INVALID_SIZE:
					throw std::runtime_error("CUFFT_INVALID_SIZE");
				case CUFFT_UNALIGNED_DATA:
					throw std::runtime_error("CUFFT_UNALIGNED_DATA");
				default:
					throw std::runtime_error("CUFFT Unknown Error");
			}
	
		}
	};
}
#endif

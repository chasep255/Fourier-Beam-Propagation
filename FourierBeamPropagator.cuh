#ifndef _FOURIER_BEAM_PROPAGATOR_
#define _FOURIER_BEAM_PROPAGATOR_

#include <cufft.h>
#include <thrust/complex.h>
#include <stdexcept>
#include <algorithm>

namespace fftbpm
{
	typedef thrust::complex<double> complex;
	static_assert(sizeof(complex) == sizeof(cufftDoubleComplex), "CUFFT complex type does not match thrust complex type.");
	
	template<typename IndexOfRefractionFunct>
	__global__ void _refraction_kernel(complex* efld, IndexOfRefractionFunct n, double dz, double z, double cell_size, double k0, size_t w, size_t h)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx;
		int idy = blockIdx.y * blockDim.y + threadIdy;
		if(idx < w && idy < h)
		{
			double x = cell_size * idx;
			double y = cell_size * idy;
			double phi = -n(x, y, z) * dz * k0;
			double re, im;
			sincos(phi, &im, &re);
			efld[idy * N + idx] *= thrust::exp(complex(re, im));
		}
	}
	
	template<typename IndexOfRefractionFunct>
	class BeamPropagator
	{
		public:
		
		BeamPropagator(size_t _w, size_t _h, 
				IndexOfRefractionFunct _n = IndexOfRefractionFunct(), double _cell_size = 1.0, double _z = 0.0) :
			w(_w), h(_h), n(_n), cell_size(_cell_size) ,z(_z)
		{
			cufftcheck(cufftPlan2d(&plan, _w, _h, CUFFT_Z2Z));
			cudaMalloc(&fft_d, sizeof(complex) * _w * _h);
			cucheck();
			
			cudaMallocHost(&efld, sizeof(complex) * _w * _h);
			cucheck();
		}
		
		~BeamPropagator()
		{
			cufftDestroy(plan);
			cudaFree(fft_d);
			cudaFreeHost(efld);
			
			plan = 0;
			fft_d = nullptr;
			efld = nullptr;
		}
		
		void setElectricField(const complex* e)
		{
			std::copy(e, e + w * h, efld);
		}
		
		complex* getElectricField()
		{
			return efld;
		}
		
		const complex* getElectricField() const
		{
			return efld;
		}
		
		size_t width() const
		{
			return w;
		}
		
		size_t height() const
		{
			return h;
		}
		
		void stepAsync(double dz, cudaStream_t stream = 0)
		{
			
		}
		
		void step(double dz, cudaStream_t stream = 0)
		{
			stepAsync(dz, stream);
			cudaStreamSynchronize(stream);
			cucheck();
		}
		
		private:
		size_t w, h;
		IndexOfRefractionFunct n;
		cufftHandle plan = 0;
		double cell_size;
		double z;
		complex* fft_d = nullptr;
		complex* efld = nullptr;
		
		dim3 refraction_block_dim;
		dim3 refraction_thread_dim;
		dim3 diffraction_block_dim;
		dim3 diffraction_thread_dim;
		
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

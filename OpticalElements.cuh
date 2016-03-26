#ifndef _OPTICAL_ELEMENTS_CUH_
#define _OPTICAL_ELEMENTS_CUH_

#include <thrust/complex.h>

template<typename real>
struct FreeSpace
{
	__device__ __host__
	thrust::complex<real> operator()(real x, real y, double z) const
	{
		return 1.0f;
	}
};

template<typename real, typename First, typename Second>
struct TwoStagePath
{
	First first;
	Second second;
	real boundary;
	
	TwoStagePath(First f, Second s, real b) :
		first(f), second(s), boundary(b) { }
	
	__device__ __host__
	thrust::complex<real> operator()(real x, real y, double z) const
	{
		if(z < boundary)
			return first(x, y, z);
		else
			return second(x, y, z);
	}
};

template<typename real, typename First, typename Second, typename Third>
struct ThreeStagePath
{
	First first;
	Second second;
	Third third;
	real boundary1, boundary2;
	
	ThreeStagePath(First f, Second s, Third t, real b1, real b2) :
		first(f), second(s), third(t), boundary1(b1), boundary2(b2) { }
	
	__device__ __host__
	thrust::complex<real> operator()(real x, real y, double z) const
	{
		if(z < boundary1)
			return first(x, y, z);
		else if(z < boundary2)
			return second(x, y, z);
		else
			return third(x, y, z);
	}
};

template<typename real, typename Path>
struct PerodicPath
{
	Path path;
	real period;
	real phi;
	
	PerodicPath(Path p, real t, real phase_shift = 0) :
		path(p), period(t), phi(phase_shift) { }
	
	__device__ __host__
	thrust::complex<real> operator()(real x, real y, double z) const
	{
		real z_prime = fmod(z + phi, period);
		return path(x, y, z_prime);
	}
};

template<typename real>
struct PlanoConvexLens
{
	real roc;
	real start;
	real end;
	real n;
	
	PlanoConvexLens(real _roc, real _start, real _end, real _n) :
		roc(_roc), start(_start), end(_end), n(_n) { }
	
	__device__ __host__
	thrust::complex<real> operator()(real x, real y, double z) const
	{
		if(z >= start && z <= end)
		{
			real dz = fabs(roc) - (roc < 0 ? end - z : z - start);
			if(x * x + y * y + dz * dz <= roc * roc)
				return n;
		}
		return 1.0f;
	}
};

template<typename real>
struct BiConvexLens
{
	real pos;
	PlanoConvexLens<real> front, back;
	
	BiConvexLens(real roc, real z, real t, real n) :
		pos(z), front(roc, z - 0.5 * t, z, n), back(-roc, z, z + 0.5 * t, n)
	{ }
	
	__device__ __host__
	thrust::complex<real> operator()(real x, real y, double z) const
	{
		if(z > pos)
			return back(x, y, z);
		else
			return front(x, y, z);
	}
};

template<typename real>
struct BinaryZonePlate
{
	real start;
	real thickness;
	real f;
	real lambda;
	thrust::complex<real> index;
	
	BinaryZonePlate(real _start, real _thickness, real focal_length, 
			real wave_length, thrust::complex<real> _n) : 
		start(_start), thickness(_thickness), f(focal_length), 
		lambda(wave_length), index(_n) { }
	
	__host__ __device__ 
	thrust::complex<real> operator()(real x, real y, double z) const
	{
		if(z >= start && z <= start + thickness)
		{			
			real r2 = x * x + y * y;
			int n = 2 * (sqrt(r2 + f * f) - f) / lambda;
			return n % 2 ? index : 1;
		}
		return 1.0;
	}
};

#endif

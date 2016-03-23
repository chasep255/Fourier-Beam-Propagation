#define CAT2(a, b) a##b
#define MAKE_VECTOR(t, n) CAT2(t, n)

typedef REAL_TYPE real;
typedef MAKE_VECTOR(REAL_TYPE, 2) real2;

#define K0 (real)(2.0 * M_PI / WAVELENGTH)
#define GRID_SIZE2 (GRID_SIZE * GRID_SIZE)

real2 cplx_mul(real2 a, real2 b)
{
	return (real2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

real2 cplx_exp(real2 a)
{
	real e = exp(a.x);
	return (real2)(e * cos(a.y), e * sin(a.y));
}

__kernel void refraction_kernel
(
	__global real* electric_field,
	__global real* nbuffer,
	real dz
)
{
	size_t id = get_global_id(0);
	if(id > GRID_SIZE2)
		return;
	
	real2 e = vload2(id, electric_field);
	#ifdef COMPLEX_N
		real2 n = vload2(id, nbuffer);
		real2 phi = cplx_exp(K0 * dz * n);
		e = cplx_mul(phi, e);
	#else
		real n = nbuffer[id];
		real phi = K0 * dz * n;
		e = cplx_mul(e, (real2)(cos(phi), sin(phi)));
	#endif
	vstore2(e, id, electric_field);
}


__kernel void diffraction_kernel
(
	__global real* electric_field_fft,
	real dz
)
{
	size_t id = get_global_id(0);
	if(id > GRID_SIZE2)
		return;
	
	int idx = id % GRID_SIZE;
	int idy = id / GRID_SIZE;
	
	real kx = 2.0 * M_PI * (idx > GRID_SIZE / 2 ? (idx - GRID_SIZE) : idx) / (real)(CELL_SIZE * GRID_SIZE);
	real ky = 2.0 * M_PI * (idy > GRID_SIZE / 2 ? (idy - GRID_SIZE) : idy) / (real)(CELL_SIZE * GRID_SIZE);
	real kz = sqrt(K0 * K0 - kx * kx - ky * ky);
	
	real2 e = vload2(id, electric_field_fft);
	real re = cos(-dz * kz);
	real im = sin(-dz * kz);
	e = cplx_mul(e, (real2)(re, im));
	vstore2(e, id, electric_field_fft);
}

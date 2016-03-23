#include "clbpm.h"
#include <stdlib.h>
#include <clFFT.h>
#include <math.h>
#include <stdio.h>

void check(int err)
{
	if(err != CLBPM_SUCCESS)
	{
		fprintf(stderr, "CLBPM ERROR CODE: %d", err);
		exit(1);
	}
}

static struct
{
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
} global;

void setupContext()
{
	cl_uint n;
	check(clGetPlatformIDs(0, NULL, &n));
	cl_platform_id* pids = alloca(sizeof(cl_platform_id) * n);
	check(clGetPlatformIDs(n, pids, NULL));
	
	puts("Select a platform...");
	for(cl_uint i = 0; i < n; i++)
	{
		size_t name_size;
		check(clGetPlatformInfo(pids[i], CL_PLATFORM_NAME, 0, NULL, &name_size));
		char* name = alloca(name_size);
		check(clGetPlatformInfo(pids[i], CL_PLATFORM_NAME, name_size, name, NULL));
		printf("%d) %s\n", i + 1, name);
	}
	printf("Enter platform number: ");
	
	unsigned int choice;
	if(scanf("%ud", &choice) != 1 || choice < 1 || choice > n)
		exit(1);
	
	global.platform = pids[choice - 1];
	
	check(clGetDeviceIDs(global.platform, CL_DEVICE_TYPE_ALL, 0, NULL, &n));
	cl_device_id* dids = alloca(sizeof(cl_device_id) * n);
	check(clGetDeviceIDs(global.platform, CL_DEVICE_TYPE_ALL, n, dids, NULL));
	
	puts("Select a device...");
	for(cl_uint i = 0; i < n; i++)
	{
		size_t name_size;
		check(clGetDeviceInfo(dids[i], CL_DEVICE_NAME, 0, NULL, &name_size));
		char* name = alloca(name_size);
		check(clGetDeviceInfo(dids[i], CL_DEVICE_NAME, name_size, name, NULL));
		printf("%d) %s\n", i + 1, name);
	}
	printf("Enter device number: ");
	
	if(scanf("%ud", &choice) != 1 || choice < 1 || choice > n)
		exit(1);
	
	global.device = dids[choice - 1];
	
	cl_int err;
	global.context = clCreateContext(NULL, 1, &global.device, NULL, NULL, &err);
	check(err);
}

void save(const char* fn, double* data)
{	
	FILE* f = fopen(fn, "w");
	fprintf(f, "P5 1024 1024 255 ");
	
	unsigned char* buffer = malloc(sizeof(char) * 1024 * 1024);
	
	double max = 0.0;
	for(int i = 0; i < 1024 * 1024; i++)
	{
		double mag = sqrt(data[2 * i] * data[2 * i] + data[2 * i + 1] * data[2 * i + 1]);
		max = fmax(mag, max);
		buffer[i] = fmin(255.0, 255.0 * mag);
	}
	printf("%lf\n", max);
	
	for(int i = 0; i < 1024 * 1024; i++)
	{
		double mag = sqrt(data[2 * i] * data[2 * i] + data[2 * i + 1] * data[2 * i + 1]);
		buffer[i] = fmin(255.0, 255.0 * mag / max);
	}
	
	fwrite(buffer, 1, 1024 * 1024, f);
	
	free(buffer);
	fclose(f);
}

int main()
{
	check(clbpmInit(1));
	setupContext();
	
	clbpm_t bpm = clbpmNew();
	check(clbpmSetCellSize(&bpm, 1.0e-6));
	check(clbpmSetGridSize(&bpm, 1024));
	check(clbpmSetFloatFormat(&bpm, CLBPM_DOUBLE));
	check(clbpmSetMaskReal(&bpm, CL_FALSE));
	check(clbpmSetWavelength(&bpm, 1.0e-6));
	check(clbpmSetContext(&bpm, global.context));
	check(clbpmCompile(&bpm));
	
	cl_int success;
	cl_command_queue cmdq = clCreateCommandQueue(global.context, global.device, 0, &success);
	check(success);
	
	cl_mem mask = clCreateBuffer(global.context, CL_MEM_READ_ONLY, sizeof(cl_double) * 1024 * 1024, NULL, &success);
	check(success);
	
	cl_double fill_pattern = 1.0;
	check(clEnqueueFillBuffer(cmdq, mask, &fill_pattern, sizeof(cl_double), 0, sizeof(cl_double) * 1024 * 1024, 0, NULL, NULL));
	
	double* initial_electric_field = clEnqueueMapBuffer(cmdq, bpm.electric_field_d, CL_TRUE, CL_MAP_WRITE, 0, 2 * sizeof(cl_double) * 1024 * 1024, 0, NULL, NULL, &success);
	check(success);
	
	for(int y = 0; y < 1024; y++)
	{
		for(int x = 0; x < 1024; x++)
		{
			initial_electric_field[2 * (y * 1024 + x) + 0] = 0.0;
			initial_electric_field[2 * (y * 1024 + x) + 1] = 0.0;
			
			int dx = x - 512;
			int dy = y - 512;
			
			if(dx * dx + dy * dy < 100 * 100)
				initial_electric_field[2 * (y * 1024 + x) + 0] = 1.0;
		}
	}
	check(clEnqueueUnmapMemObject(cmdq, bpm.electric_field_d, initial_electric_field, 0, NULL, NULL));
	
	for(int i = 0; i < 20; i++)
		check(clbpmStep(&bpm, cmdq, 1.0e-6, mask, 0, NULL, NULL));
	
	double* stepped = clEnqueueMapBuffer(cmdq, bpm.electric_field_d, CL_TRUE, CL_MAP_READ, 0, 2 * sizeof(cl_double) * 1024 * 1024, 0, NULL, NULL, &success);
	check(success);
	
	save("test.ppm", stepped);
	
	check(clFinish(cmdq));
	check(clbpmTeardown(1));
	return 0;
}

#include "clbpm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static const char* const CLBPM_SOURCE_FILE = "/home/chase/workspace/FFTBPM/clsrc/bpm.cl";
static char* clbpm_source_code = NULL;
static size_t clbpm_source_length = 0;

cl_int clbpmInit(int init_clfft)
{
	FILE* f = NULL;
	
	cl_int status = CLBPM_SUCCESS;
	
	if(clbpm_source_code)
		return CLBPM_ALREADY_INITIALIZED;
	
	f = fopen(CLBPM_SOURCE_FILE, "r");
	if(f == NULL)
	{
		status = CLBPM_FILE_NOT_FOUND;
		goto cleanup;
	}
	
	if(fseek(f, 0, SEEK_END))
	{
		status = CLBPM_IO_ERROR;
		goto cleanup;
	}
	
	clbpm_source_length = ftell(f);
	
	if(fseek(f, 0, SEEK_SET))
	{
		status = CLBPM_IO_ERROR;
		goto cleanup;
	}
	
	clbpm_source_code = malloc(sizeof(char) * clbpm_source_length);
	if(fread(clbpm_source_code, sizeof(char), clbpm_source_length, f) != clbpm_source_length)
	{
		status = CLBPM_IO_ERROR;
		goto cleanup;
	}
	
	if(init_clfft)
	{
		status = clfftSetup(NULL);
		if(status != CLBPM_SUCCESS)
			goto cleanup;
	}
	
	return status;
	cleanup:
	if(f) fclose(f);
	if(clbpm_source_code) free(clbpm_source_code);
	return status;
}

cl_int clbpmTeardown(int teardown_clfft)
{
	if(clbpm_source_code)
		free(clbpm_source_code);
	clbpm_source_code = NULL;
	clbpm_source_length = 0;
	if(teardown_clfft)
		return clfftTeardown();
	
	return CLBPM_SUCCESS;
}

clbpm_t clbpmNew()
{
	clbpm_t bpm = { };
	return bpm;
}

cl_int clbpmCompile(clbpm_t* bpm)
{
	if(bpm == NULL)
		return CLBPM_NULL_POINTER;
	if(bpm->cell_size <= 0 || bpm->grid_size <= 0 || bpm->wavelength <= 0)
		return CLBPM_INVALID_CONFIGURATION;
	if(bpm->compiled)
		return CLBPM_ALREADY_COMPILED;
	
	cl_int status = CLBPM_SUCCESS;
	
	size_t grid_mem_size = 2 * clbpmSizeofFloat(bpm->float_type) * bpm->grid_size * bpm->grid_size;
	bpm->electric_field_d = clCreateBuffer(bpm->ctx, CL_MEM_READ_WRITE, grid_mem_size, NULL, &status);
	if(status != CLBPM_SUCCESS)
		goto cleanup;
	
	bpm->electric_field_fft_d = clCreateBuffer(bpm->ctx, CL_MEM_READ_WRITE, grid_mem_size, NULL, &status);
	if(status != CLBPM_SUCCESS)
		goto cleanup;
	
	bpm->pgrm = clCreateProgramWithSource(bpm->ctx, 1, (const char**)&clbpm_source_code, &clbpm_source_length, &status);
	if(status != CLBPM_SUCCESS)
		goto cleanup;
	
	char build_args[1024] = "-cl-fast-relaxed-math ";
	if(bpm->float_type == CLBPM_FLOAT)
		strcat(build_args, "-DREAL_TYPE=float -cl-single-precision-constant ");
	else
		strcat(build_args, "-DREAL_TYPE=double ");
	sprintf(build_args + strlen(build_args), "-DGRID_SIZE=%zd -DCELL_SIZE=%0.30lf -DWAVELENGTH=%0.30lf ", 
			bpm->grid_size, bpm->cell_size, bpm->wavelength);
	if(bpm->mask_real)
		strcat(build_args, "-DCOMPLEX_N ");
	
	status = clBuildProgram(bpm->pgrm, 0, NULL, build_args, NULL, NULL);
	if(status != CL_SUCCESS)
	{
		size_t ndevs;
		status = clGetProgramInfo(bpm->pgrm, CL_PROGRAM_NUM_DEVICES, sizeof(size_t), &ndevs, NULL);
		if(status != CLBPM_SUCCESS)
			goto cleanup;
		cl_device_id* devs = alloca(sizeof(cl_device_id) * ndevs);
		status = clGetProgramInfo(bpm->pgrm, CL_PROGRAM_DEVICES, sizeof(cl_device_id) * ndevs, devs, NULL);
		if(status != CLBPM_SUCCESS)
			goto cleanup;
		
		for(int d = 0; d < ndevs; d++)
		{
			cl_build_status build_status;
			status = clGetProgramBuildInfo(bpm->pgrm, devs[d], CL_PROGRAM_BUILD_STATUS, 
					sizeof(cl_build_status), &build_status, NULL);
			if(status != CLBPM_SUCCESS)
				goto cleanup;
			if(build_status == CL_BUILD_ERROR)
			{
				size_t len;
				status = clGetProgramBuildInfo(bpm->pgrm, devs[d], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
				if(status != CLBPM_SUCCESS)
					goto cleanup;
				char* error_message = alloca(len);
				status = clGetProgramBuildInfo(bpm->pgrm, devs[d], CL_PROGRAM_BUILD_LOG, len, error_message, NULL);
				if(status != CLBPM_SUCCESS)
					goto cleanup;
				
				status = clGetDeviceInfo(devs[d], CL_DEVICE_NAME, 0, NULL, &len);
				if(status != CLBPM_SUCCESS)
					goto cleanup;
				
				char* device_name = alloca(len);
				status = clGetDeviceInfo(devs[d], CL_DEVICE_NAME, len, device_name, NULL);
				if(status != CLBPM_SUCCESS)
					goto cleanup;
				
				fprintf(stderr, "%s\n", device_name);
				fputs(error_message, stderr);
			}
		}
		goto cleanup;
	}
	
	bpm->refraction_kernel = clCreateKernel(bpm->pgrm, "refraction_kernel", &status);
	if(status != CLBPM_SUCCESS)
		goto cleanup;
	
	bpm->diffraction_kernel = clCreateKernel(bpm->pgrm, "diffraction_kernel", &status);
	if(status != CLBPM_SUCCESS)
		goto cleanup;
	
	size_t plan_lengths[2] = {bpm->grid_size, bpm->grid_size};
	status = clfftCreateDefaultPlan(&(bpm->fft_plan), bpm->ctx, 2, plan_lengths);
	if(status != CLBPM_SUCCESS)
		goto cleanup;
	
	status = clfftSetPlanPrecision(bpm->fft_plan, bpm->float_type);
	if(status != CLBPM_SUCCESS)
		goto cleanup;
	
	status = clfftSetLayout(bpm->fft_plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
	if(status != CLBPM_SUCCESS)
		goto cleanup;
	
	status = clfftSetPlanScale(bpm->fft_plan, CLFFT_FORWARD, 1.0f / (bpm->grid_size * bpm->grid_size));
	if(status != CLBPM_SUCCESS)
		goto cleanup;
	
	status = clfftSetPlanScale(bpm->fft_plan, CLFFT_BACKWARD, 1.0f);
	if(status != CLBPM_SUCCESS)
		goto cleanup;
	
	status = clfftSetResultLocation(bpm->fft_plan, CLFFT_OUTOFPLACE);
	if(status != CLBPM_SUCCESS)
		goto cleanup;
	
	return CLBPM_SUCCESS;
	cleanup:
	clbpmFree(bpm);
	return status;
}

cl_int clbpmFree(clbpm_t* bpm)
{
	clReleaseMemObject(bpm->electric_field_d);
	clReleaseMemObject(bpm->electric_field_fft_d);
	clReleaseKernel(bpm->diffraction_kernel);
	clReleaseKernel(bpm->refraction_kernel);
	clReleaseProgram(bpm->pgrm);
	clfftDestroyPlan(&(bpm->fft_plan));
	memset(bpm, 0, sizeof(clbpm_t));
	return CLBPM_SUCCESS;
}

cl_int clbpmStep(clbpm_t* bpm, cl_command_queue cmdq, double dz, cl_mem mask, 
		cl_uint num_events, const cl_event* wait_list, cl_event* event)
{
	cl_int status = CLBPM_SUCCESS;
	
	status = clSetKernelArg(bpm->refraction_kernel, 0, sizeof(cl_mem), &(bpm->electric_field_d));
	if(status != CLBPM_SUCCESS)
		goto end;
	
	status = clSetKernelArg(bpm->refraction_kernel, 1, sizeof(cl_mem), &mask);
	if(status != CLBPM_SUCCESS)
		goto end;
	
	float dzf = dz;
	status = clSetKernelArg(bpm->refraction_kernel, 2, clbpmSizeofFloat(bpm->float_type),
			bpm->float_type == CLBPM_FLOAT ? (void*)&dzf : (void*)&dz);
	if(status != CLBPM_SUCCESS)
		goto end;
	
	status = clSetKernelArg(bpm->diffraction_kernel, 0, sizeof(cl_mem), &(bpm->electric_field_fft_d));
	if(status != CLBPM_SUCCESS)
		goto end;
	
	status = clSetKernelArg(bpm->diffraction_kernel, 1, clbpmSizeofFloat(bpm->float_type),
			bpm->float_type == CLBPM_FLOAT ? (void*)&dzf : (void*)&dz);
	if(status != CLBPM_SUCCESS)
		goto end;
	
	size_t work_items = bpm->grid_size * bpm->grid_size;
	cl_event refraction_finished;
	status = clEnqueueNDRangeKernel(cmdq, bpm->refraction_kernel, 1, NULL, &work_items, 
			NULL, num_events, wait_list, &refraction_finished);
	if(status != CLBPM_SUCCESS)
		goto end;

	cl_event forward_finished;
	status = clfftEnqueueTransform(bpm->fft_plan, CLFFT_FORWARD, 1, &cmdq, 1, &refraction_finished, 
			&forward_finished, &(bpm->electric_field_d), &(bpm->electric_field_fft_d), 0);
	if(status != CLBPM_SUCCESS)
		goto end;
	
	cl_event diffraction_finished;
	status = clEnqueueNDRangeKernel(cmdq, bpm->diffraction_kernel, 1, NULL, &work_items, NULL, 1, 
			&forward_finished, &diffraction_finished);
	if(status != CLBPM_SUCCESS)
		goto end;
	
	status = clfftEnqueueTransform(bpm->fft_plan, CLFFT_BACKWARD, 1, &cmdq, 1, &refraction_finished, 
			event, &(bpm->electric_field_fft_d), &(bpm->electric_field_d), 0);
	if(status != CLBPM_SUCCESS)
		goto end;
	
	end:
	return status;
}

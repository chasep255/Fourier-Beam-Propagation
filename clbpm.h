#ifndef _CL_BPM_H_
#define _CL_BPM_H_

#include <CL/cl.h>
#include <clFFT.h>

typedef enum
{
	CLBPM_INVALID_GLOBAL_WORK_SIZE			= CL_INVALID_GLOBAL_WORK_SIZE,
	CLBPM_INVALID_MIP_LEVEL					= CL_INVALID_MIP_LEVEL,
	CLBPM_INVALID_BUFFER_SIZE				= CL_INVALID_BUFFER_SIZE,
	CLBPM_INVALID_GL_OBJECT					= CL_INVALID_GL_OBJECT,
	CLBPM_INVALID_OPERATION					= CL_INVALID_OPERATION,
	CLBPM_INVALID_EVENT						= CL_INVALID_EVENT,
	CLBPM_INVALID_EVENT_WAIT_LIST			= CL_INVALID_EVENT_WAIT_LIST,
	CLBPM_INVALID_GLOBAL_OFFSET				= CL_INVALID_GLOBAL_OFFSET,
	CLBPM_INVALID_WORK_ITEM_SIZE			= CL_INVALID_WORK_ITEM_SIZE,
	CLBPM_INVALID_WORK_GROUP_SIZE			= CL_INVALID_WORK_GROUP_SIZE,
	CLBPM_INVALID_WORK_DIMENSION			= CL_INVALID_WORK_DIMENSION,
	CLBPM_INVALID_KERNEL_ARGS				= CL_INVALID_KERNEL_ARGS,
	CLBPM_INVALID_ARG_SIZE					= CL_INVALID_ARG_SIZE,
	CLBPM_INVALID_ARG_VALUE					= CL_INVALID_ARG_VALUE,
	CLBPM_INVALID_ARG_INDEX					= CL_INVALID_ARG_INDEX,
	CLBPM_INVALID_KERNEL					= CL_INVALID_KERNEL,
	CLBPM_INVALID_KERNEL_DEFINITION			= CL_INVALID_KERNEL_DEFINITION,
	CLBPM_INVALID_KERNEL_NAME				= CL_INVALID_KERNEL_NAME,
	CLBPM_INVALID_PROGRAM_EXECUTABLE		= CL_INVALID_PROGRAM_EXECUTABLE,
	CLBPM_INVALID_PROGRAM					= CL_INVALID_PROGRAM,
	CLBPM_INVALID_BUILD_OPTIONS				= CL_INVALID_BUILD_OPTIONS,
	CLBPM_INVALID_BINARY					= CL_INVALID_BINARY,
	CLBPM_INVALID_SAMPLER					= CL_INVALID_SAMPLER,
	CLBPM_INVALID_IMAGE_SIZE				= CL_INVALID_IMAGE_SIZE,
	CLBPM_INVALID_IMAGE_FORMAT_DESCRIPTOR	= CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
	CLBPM_INVALID_MEM_OBJECT				= CL_INVALID_MEM_OBJECT,
	CLBPM_INVALID_HOST_PTR					= CL_INVALID_HOST_PTR,
	CLBPM_INVALID_COMMAND_QUEUE				= CL_INVALID_COMMAND_QUEUE,
	CLBPM_INVALID_QUEUE_PROPERTIES			= CL_INVALID_QUEUE_PROPERTIES,
	CLBPM_INVALID_CONTEXT					= CL_INVALID_CONTEXT,
	CLBPM_INVALID_DEVICE					= CL_INVALID_DEVICE,
	CLBPM_INVALID_PLATFORM					= CL_INVALID_PLATFORM,
	CLBPM_INVALID_DEVICE_TYPE				= CL_INVALID_DEVICE_TYPE,
	CLBPM_INVALID_VALUE						= CL_INVALID_VALUE,
	CLBPM_MAP_FAILURE						= CL_MAP_FAILURE,
	CLBPM_BUILD_PROGRAM_FAILURE				= CL_BUILD_PROGRAM_FAILURE,
	CLBPM_IMAGE_FORMAT_NOT_SUPPORTED		= CL_IMAGE_FORMAT_NOT_SUPPORTED,
	CLBPM_IMAGE_FORMAT_MISMATCH				= CL_IMAGE_FORMAT_MISMATCH,
	CLBPM_MEM_COPY_OVERLAP					= CL_MEM_COPY_OVERLAP,
	CLBPM_PROFILING_INFO_NOT_AVAILABLE		= CL_PROFILING_INFO_NOT_AVAILABLE,
	CLBPM_OUT_OF_HOST_MEMORY				= CL_OUT_OF_HOST_MEMORY,
	CLBPM_OUT_OF_RESOURCES					= CL_OUT_OF_RESOURCES,
	CLBPM_MEM_OBJECT_ALLOCATION_FAILURE		= CL_MEM_OBJECT_ALLOCATION_FAILURE,
	CLBPM_COMPILER_NOT_AVAILABLE			= CL_COMPILER_NOT_AVAILABLE,
	CLBPM_DEVICE_NOT_AVAILABLE				= CL_DEVICE_NOT_AVAILABLE,
	CLBPM_DEVICE_NOT_FOUND					= CL_DEVICE_NOT_FOUND,
	CLBPM_SUCCESS							= CL_SUCCESS,

	CLBPM_BUGCHECK                          = CLFFT_BUGCHECK,	
	CLBPM_NOTIMPLEMENTED                    = CLFFT_NOTIMPLEMENTED,		
	CLBPM_TRANSPOSED_NOTIMPLEMENTED         = CLFFT_TRANSPOSED_NOTIMPLEMENTED, 
	CLBPM_FILE_NOT_FOUND                    = CLFFT_FILE_NOT_FOUND,		
	CLBPM_FILE_CREATE_FAILURE               = CLFFT_FILE_CREATE_FAILURE,	
	CLBPM_VERSION_MISMATCH                  = CLFFT_VERSION_MISMATCH,		
	CLBPM_INVALID_PLAN                      = CLFFT_INVALID_PLAN,			
	CLBPM_DEVICE_NO_DOUBLE                  = CLFFT_DEVICE_NO_DOUBLE,		
	CLBPM_DEVICE_MISMATCH                   = CLFFT_DEVICE_MISMATCH,
	CLBMP_ENDSTATUS                         = CLFFT_ENDSTATUS,
	
	CLBPM_NULL_POINTER,
	CLBPM_ALREADY_COMPILED,
	CLBPM_IO_ERROR,
	CLBPM_ALREADY_INITIALIZED,
	CLBPM_INVALID_CONFIGURATION
} clbpm_status_t;

typedef enum
{
	CLBPM_DOUBLE = CLFFT_DOUBLE,
	CLBPM_FLOAT = CLFFT_SINGLE
} clbpm_float_type_t;

typedef struct
{
	clbpm_float_type_t float_type;
	cl_bool mask_real;
	cl_bool compiled;
	size_t grid_size;
	cl_double cell_size;
	cl_double wavelength;
	clfftPlanHandle fft_plan;
	cl_context ctx;
	cl_program pgrm;
	cl_kernel refraction_kernel;
	cl_kernel diffraction_kernel;
	cl_mem electric_field_d;
	cl_mem electric_field_fft_d;
} clbpm_t;

cl_int clbpmInit(int init_clfft);
cl_int clbpmTeardown(int teardown_clfft);

clbpm_t clbpmNew();
cl_int clbpmCompile(clbpm_t* bpm);
cl_int clbpmFree(clbpm_t* bpm);
cl_int clbpmStep(clbpm_t* bpm, cl_command_queue cmdq, cl_double dz, cl_mem mask, 
		cl_uint num_events, const cl_event* wait_list, cl_event* event);

static inline clbpm_status_t clbpmSetWavelength(clbpm_t* bpm, cl_double wavelength)
{
	if(bpm == NULL)
		return CLBPM_NULL_POINTER;
	if(bpm->compiled)
		return CLBPM_ALREADY_COMPILED;
	if(wavelength <= 0.0)
		return CLBPM_INVALID_VALUE;
	
	bpm->wavelength = wavelength;
	
	return CLBPM_SUCCESS;
}

static inline clbpm_status_t clbpmSetCellSize(clbpm_t* bpm, cl_double cell_size)
{
	if(bpm == NULL)
		return CLBPM_NULL_POINTER;
	if(bpm->compiled)
		return CLBPM_ALREADY_COMPILED;
	if(cell_size <= 0.0)
		return CLBPM_INVALID_VALUE;
	
	bpm->cell_size = cell_size;
	
	return CLBPM_SUCCESS;
}

static inline clbpm_status_t clbpmSetContext(clbpm_t* bpm, cl_context ctx)
{
	if(bpm == NULL)
		return CLBPM_NULL_POINTER;
	if(bpm->compiled)
		return CLBPM_ALREADY_COMPILED;
	
	bpm->ctx = ctx;
	
	return CLBPM_SUCCESS;
}

static inline clbpm_status_t clbpmSetGridSize(clbpm_t* bpm, int grid_size)
{
	if(bpm == NULL)
		return CLBPM_NULL_POINTER;
	if(bpm->compiled)
		return CLBPM_ALREADY_COMPILED;
	if(grid_size <= 0)
		return CLBPM_INVALID_VALUE;
	
	bpm->grid_size = grid_size;
	
	return CLBPM_SUCCESS;
}

static inline clbpm_status_t clbpmSetMaskReal(clbpm_t* bpm, cl_bool mask_real)
{
	if(bpm == NULL)
		return CLBPM_NULL_POINTER;
	if(bpm->compiled)
		return CLBPM_ALREADY_COMPILED;
	
	bpm->mask_real = mask_real;
	
	return CLBPM_SUCCESS;
}

static inline clbpm_status_t clbpmSetFloatFormat(clbpm_t* bpm, clbpm_float_type_t float_type)
{
	if(bpm == NULL)
		return CLBPM_NULL_POINTER;
	if(bpm->compiled)
		return CLBPM_ALREADY_COMPILED;
	
	bpm->float_type = float_type;
	
	return CLBPM_SUCCESS;
}

static inline cl_bool clbpmGetMaskReal(clbpm_t* bpm)
{
	return bpm->mask_real;
}

static inline clbpm_float_type_t clbpmGetElectricFieldFormat(clbpm_t* bpm)
{
	return bpm->float_type;
}

static inline cl_double clbpmGetWavelength(clbpm_t* bpm)
{
	return bpm->wavelength;
}

static inline cl_double clbpmGetCellSize(clbpm_t* bpm)
{
	return bpm->cell_size;
}

static inline size_t clbpmGetGridSize(clbpm_t* bpm)
{
	return bpm->grid_size;
}

static inline cl_context clbpmGetContext(clbpm_t* bpm)
{
	return bpm->ctx;
}

static inline cl_mem clbpmGetElectricFieldBuffer(clbpm_t* bpm)
{
	return bpm->electric_field_d;
}

static inline cl_bool clbpmIsCompiled(clbpm_t* bpm)
{
	return bpm->compiled;
}

static inline size_t clbpmSizeofFloat(clbpm_float_type_t float_type)
{
	switch(float_type)
	{
		case CLBPM_FLOAT: return sizeof(cl_float);
		case CLBPM_DOUBLE: return sizeof(cl_double);
	}
	return 0;
}

#endif

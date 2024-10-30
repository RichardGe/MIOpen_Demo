
//
//
// ILLUSTRATE SIMPLE TENSOR CONVOLUTION WITH MIOPEN/HIP
//
//

#include <iostream>
#include <vector>
#include <miopen/miopen.h>
#include <hip/hip_runtime.h>

#define CHECK_MIOPEN_ERROR(expression)					   \
	{														\
		miopenStatus_t status = (expression);				\
		if (status != miopenStatusSuccess) {				 \
			std::cerr << "Error on line " << __LINE__ << ": " \
					  << miopenGetErrorString(status) << "\n";\
			exit(EXIT_FAILURE);							  \
		}													\
	}

#define CHECK_HIP_ERROR(expression)						  \
	{														\
		hipError_t status = (expression);					\
		if (status != hipSuccess) {						  \
			std::cerr << "HIP Error on line " << __LINE__ << ": "\
					  << hipGetErrorString(status) << "\n";  \
			exit(EXIT_FAILURE);							  \
		}													\
	}

void printTensor(const std::vector<float>& tensor, int n, int c, int h, int w) {
	for (int ni = 0; ni < n; ++ni) {
		for (int ci = 0; ci < c; ++ci) {
			for (int hi = 0; hi < h; ++hi) {
				for (int wi = 0; wi < w; ++wi) {
					std::cout << tensor[ni * c * h * w + ci * h * w + hi * w + wi] << " ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
	}
}



int main() 
{
	// Initialize HIP and check for GPUs
	int deviceCount = 0;
	CHECK_HIP_ERROR(hipGetDeviceCount(&deviceCount));
	if (deviceCount == 0) {
		std::cerr << "No GPU device found. Exiting.\n";
		return EXIT_FAILURE;
	}

	// Select GPU device 0
	hipDeviceProp_t deviceProp;
	CHECK_HIP_ERROR(hipGetDeviceProperties(&deviceProp, 0));
	if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
		std::cerr << "Selected device 0 is not a valid GPU. Exiting.\n";
		return EXIT_FAILURE;
	}
	std::cout << "Using GPU device: " << deviceProp.name << std::endl;
	CHECK_HIP_ERROR(hipSetDevice(0));

	// Initialize MIOpen
	miopenHandle_t handle;
	CHECK_MIOPEN_ERROR(miopenCreate(&handle));

	// Set up tensor descriptors
	miopenTensorDescriptor_t inputTensor, outputTensor, filterTensor;
	CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(&inputTensor));
	CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(&outputTensor));
	CHECK_MIOPEN_ERROR(miopenCreateTensorDescriptor(&filterTensor));

	const int n = 1, c = 1, h = 5, w = 5;
	const int k = 1, r = 3, s = 3;

	CHECK_MIOPEN_ERROR(miopenSet4dTensorDescriptor(inputTensor, miopenFloat, n, c, h, w));
	CHECK_MIOPEN_ERROR(miopenSet4dTensorDescriptor(outputTensor, miopenFloat, n, k, h-2, w-2));
	CHECK_MIOPEN_ERROR(miopenSet4dTensorDescriptor(filterTensor, miopenFloat, k, c, r, s));

	// Allocate memory on the device
	std::vector<float> outputData(n * k * (h-2) * (w-2), 0.0f);
	
	std::vector<float> inputData = {  
		0.0f, 1.0f, 1.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
		0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
		0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
		0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
	};
	assert( inputData.size() == n * c * h * w);
	

	std::vector<float> filterData = {  
		1.0f, 0.0f, 1.0f,
		0.0f, 1.0f, 0.0f,
		1.0f, 0.0f, 1.0f,
	};
	assert( filterData.size() == k * c * r * s);
	
	// expected result of convolution:
	// 1 4 3
	// 1 2 4
	// 1 2 3
	

	float *d_input, *d_output, *d_filter;
	CHECK_HIP_ERROR(hipMalloc(&d_input, n * c * h * w * sizeof(float)));
	CHECK_HIP_ERROR(hipMalloc(&d_output, n * k * (h-2) * (w-2) * sizeof(float)));
	CHECK_HIP_ERROR(hipMalloc(&d_filter, k * c * r * s * sizeof(float)));

	// Copy data to the GPU
	CHECK_HIP_ERROR(hipMemcpy(d_input, inputData.data(), n * c * h * w * sizeof(float), hipMemcpyHostToDevice));
	CHECK_HIP_ERROR(hipMemcpy(d_filter, filterData.data(), k * c * r * s * sizeof(float), hipMemcpyHostToDevice));

	// Initialize convolution descriptor
	miopenConvolutionDescriptor_t convDesc;
	CHECK_MIOPEN_ERROR(miopenCreateConvolutionDescriptor(&convDesc));
	CHECK_MIOPEN_ERROR(miopenInitConvolutionDescriptor(convDesc, miopenConvolution, 0, 0, 1, 1, 1, 1));

	// Allocate workspace for convolution
	size_t workspaceSize = 0;
	CHECK_MIOPEN_ERROR(miopenConvolutionForwardGetWorkSpaceSize(handle, filterTensor, inputTensor, convDesc, outputTensor, &workspaceSize));

	void *d_workspace;
	CHECK_HIP_ERROR(hipMalloc(&d_workspace, workspaceSize));

	// Choose a valid convolution algorithm
	miopenConvFwdAlgorithm_t algo;
	int retAlgoCount;
	miopenConvAlgoPerf_t perfResults;
	CHECK_MIOPEN_ERROR(miopenFindConvolutionForwardAlgorithm(handle, inputTensor, d_input, filterTensor, d_filter, convDesc, outputTensor, d_output, 1, &retAlgoCount, &perfResults, d_workspace, workspaceSize, false));
	algo = perfResults.fwd_algo;

	// Convolution forward
	float alpha = 1.0f, beta = 0.0f;
	CHECK_MIOPEN_ERROR(miopenConvolutionForward(handle, &alpha, inputTensor, d_input, filterTensor, d_filter, convDesc, algo, &beta, outputTensor, d_output, d_workspace, workspaceSize));

	// Synchronize to ensure computation is done on GPU
	CHECK_HIP_ERROR(hipDeviceSynchronize());

	// Copy output results back to the CPU
	CHECK_HIP_ERROR(hipMemcpy(outputData.data(), d_output, n * k * (h-2) * (w-2) * sizeof(float), hipMemcpyDeviceToHost));

	// Print results
	std::cout << "Input Tensor:" << std::endl;
	printTensor(inputData, n, c, h, w);

	std::cout << "Filter Tensor:" << std::endl;
	printTensor(filterData, k, c, r, s);

	std::cout << "Output Tensor:" << std::endl;
	printTensor(outputData, n, k, h-2, w-2);

	// Cleanup
	CHECK_HIP_ERROR(hipFree(d_input));
	CHECK_HIP_ERROR(hipFree(d_output));
	CHECK_HIP_ERROR(hipFree(d_filter));
	CHECK_HIP_ERROR(hipFree(d_workspace));

	CHECK_MIOPEN_ERROR(miopenDestroyTensorDescriptor(inputTensor));
	CHECK_MIOPEN_ERROR(miopenDestroyTensorDescriptor(outputTensor));
	CHECK_MIOPEN_ERROR(miopenDestroyTensorDescriptor(filterTensor));
	CHECK_MIOPEN_ERROR(miopenDestroyConvolutionDescriptor(convDesc));
	CHECK_MIOPEN_ERROR(miopenDestroy(handle));

	std::cout << "Convolution operation completed successfully on the GPU." << std::endl;

	return 0;
}



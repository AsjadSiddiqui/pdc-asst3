#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// Kernel for performing the addition step in the upsweep phase
__global__ void upsweep_op_kernel(int* scan_array, int offset, int stride_val, int array_length) {
    // Get the thread's position in the grid
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Calculate the position in the array this thread will process
    unsigned int pos = tid * stride_val;

    // Calculate indices for the operation
    unsigned int left_idx = pos + offset - 1;
    unsigned int right_idx = pos + stride_val - 1;

    // Make sure we don't go out of bounds
    if (left_idx < array_length && right_idx < array_length) {
        // Add the left element to the right element
        scan_array[right_idx] += scan_array[left_idx];
    }
}

// Kernel for performing the swap and add step in the downsweep phase
__global__ void downsweep_op_kernel(int* scan_array, int offset, int stride_val, int array_length) {
    // Get the thread's position in the grid
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Calculate the position in the array this thread will process
    unsigned int pos = tid * stride_val;

    // Calculate indices for the operation
    unsigned int left_idx = pos + offset - 1;
    unsigned int right_idx = pos + stride_val - 1;

    // Make sure we don't go out of bounds
    if (left_idx < array_length && right_idx < array_length) {
        // Perform swap and add operation
        int temp = scan_array[left_idx];
        scan_array[left_idx] = scan_array[right_idx];
        scan_array[right_idx] += temp;
    }
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result) {
    // If input and result are different pointers, copy input to result
    if (input != result) {
        cudaMemcpy(result, input, N * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    // Round N up to the next power of 2 for the algorithm
    int rounded_length = nextPow2(N);

    // Initialize variables for the upsweep phase
    int offset = 1;
    int max_offset = rounded_length / 2;

    // Execute upsweep phase with a for loop
    for (; offset <= max_offset; offset *= 2) {
        // Calculate the stride between elements
        int stride = offset * 2;

        // Calculate the number of work items based on rounded length
        int work_items = rounded_length / stride;
        if (work_items * stride < rounded_length) work_items++;

        // Ensure we have a minimum of 1 work item
        work_items = (work_items > 0) ? work_items : 1;

        // Calculate grid dimensions
        int block_count = (work_items + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // Launch the kernel with enough threads to handle rounded_length
        upsweep_op_kernel<<<block_count, THREADS_PER_BLOCK>>>(result, offset, stride, rounded_length);

        // Wait for kernel to complete
        cudaDeviceSynchronize();
    }

    // Set the last element to 0 (identity element for addition)
    int identity = 0;
    cudaMemcpy(&result[rounded_length - 1], &identity, sizeof(int), cudaMemcpyHostToDevice);

    // Execute downsweep phase with a while loop
    offset = rounded_length / 2;
    while (offset >= 1) {
        // Calculate the stride between elements
        int stride = offset * 2;

        // Calculate the number of work items based on rounded length
        int work_items = rounded_length / stride;
        if (work_items * stride < rounded_length) work_items++;

        // Ensure we have a minimum of 1 work item
        work_items = (work_items > 0) ? work_items : 1;

        // Calculate grid dimensions
        int block_count = (work_items + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // Launch the kernel with enough threads to handle rounded_length
        downsweep_op_kernel<<<block_count, THREADS_PER_BLOCK>>>(result, offset, stride, rounded_length);

        // Wait for kernel to complete
        cudaDeviceSynchronize();

        // Divide offset by 2 for next iteration
        offset /= 2;
    }
}

//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray) {
    int* device_result;
    int* device_input;
    int N = end - inarray;

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);

    cudaMalloc((void**)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void**)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration;
}

// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {
    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration;
}

// Kernel that marks positions where adjacent elements are equal
__global__ void mark_adjacent_equals(int* input_array, int* flag_array, int array_len) {
    // Calculate thread's global index
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process elements within array bounds
    if (thread_idx < array_len - 1) {
        // Compare current element with next element
        bool is_repeat = (input_array[thread_idx] == input_array[thread_idx + 1]);
        // Set flag based on comparison result
        if (is_repeat) {
            flag_array[thread_idx] = 1;
        } else {
            flag_array[thread_idx] = 0;
        }
    } else if (thread_idx == array_len - 1) {
        // Last element can't be the start of a repeat
        flag_array[thread_idx] = 0;
    }
}

// Kernel that builds the output array of repeat indices
__global__ void extract_repeat_indices(int* flag_array, int* scan_result, int* output_array, int array_len) {
    // Calculate thread's global index
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process valid elements that are marked as repeats
    if (thread_idx < array_len - 1) {
        if (flag_array[thread_idx] == 1) {
            // Place this index at its computed position in the output array
            output_array[scan_result[thread_idx]] = thread_idx;
        }
    }
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {
    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    // Store size of int to use consistently throughout the code
    const size_t int_size = sizeof(int);

    // Initialize counters and temporary values
    int repeat_count = 0;
    int last_flag_value = 0;
    int last_scan_value = 0;

    // Allocate device memory for flags array (will store 1s where repeats occur)
    int* device_flags = NULL;
    cudaMalloc((void**)&device_flags, length * int_size);

    // Initialize the flags array to zeros using cudaMemset
    cudaMemset(device_flags, 0, length * int_size);

    // Calculate kernel launch parameters for efficient execution
    int threads_per_block = THREADS_PER_BLOCK;
    int num_blocks = (length + threads_per_block - 1) / threads_per_block;
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(num_blocks);

    // Step 1: Launch kernel to mark repeating elements
    mark_adjacent_equals<<<grid_dim, block_dim>>>(device_input, device_flags, length);

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // Allocate device memory for scan results
    int* device_scan = NULL;
    cudaMalloc((void**)&device_scan, length * int_size);

    // Step 2: Perform exclusive scan on flags to determine output positions
    // Uses the exclusive_scan function that was implemented earlier
    exclusive_scan(device_flags, length, device_scan);

    // Step 3: Calculate total number of repeats by reading scan results
    if (length > 0) {
        // Copy last flag and last scan value from device to host
        cudaMemcpy(&last_flag_value, &device_flags[length - 1], int_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_scan_value, &device_scan[length - 1], int_size, cudaMemcpyDeviceToHost);

        // Total repeats is the sum of the last scan value and the last flag
        repeat_count = last_scan_value + last_flag_value;
    }

    // Step 4: Only process output array if we found repeats
    if (repeat_count > 0) {
        // Launch kernel to extract the indices of repeats
        extract_repeat_indices<<<grid_dim, block_dim>>>(
            device_flags, device_scan, device_output, length);

        // Wait for kernel to complete
        cudaDeviceSynchronize();
    }

    // Step 5: Clean up temporary device memory
    cudaFree(device_flags);
    cudaFree(device_scan);

    // Return the total count of repeats found
    return repeat_count;
}

//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int* input, int length, int* output, int* output_length) {
    int* device_input;
    int* device_output;
    int rounded_length = nextPow2(length);

    cudaMalloc((void**)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void**)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();

    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime;
    return duration;
}

void printCudaInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

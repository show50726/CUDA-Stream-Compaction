#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernScan(int n, int* odata, int* idata, int offset) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx >= n)
                return;

            if (idx >= offset) {
                odata[idx] = idata[idx] + idata[idx - offset];
            }
            else {
                odata[idx] = idata[idx];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int sizeInBytes = n * sizeof(int);
            int* dev_odata;
            int* dev_idata;
            cudaMalloc((void**)&dev_odata, sizeInBytes);
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, sizeInBytes);
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, sizeInBytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_idata failed!");

            int blockSize = 256;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            
            int iteration = ilog2ceil(n);
            int offset = 1;
            for (int i = 1; i <= iteration; i++) {
                kernScan << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, offset);
                checkCUDAError("kernScan executed failed!");
                offset <<= 1;
                std::swap(dev_odata, dev_idata);
            }

            timer().endGpuTimer();

            // We swap the buffer at the end of the loop, so the target value would be dev_idata
            cudaMemcpy(odata + 1, dev_idata, (sizeInBytes - sizeof(int)), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            // cleanup
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            checkCUDAError("cudaFree failed!");
        }
    }
}

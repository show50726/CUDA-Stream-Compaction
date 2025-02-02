#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n)(((n) >> NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)))

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_ELEMENTS_PER_BLOCK (MAX_THREADS_PER_BLOCK * 2)
#define MAX_SHARE_SIZE (MAX_ELEMENTS_PER_BLOCK + CONFLICT_FREE_OFFSET(MAX_ELEMENTS_PER_BLOCK - 1))

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        using StreamCompaction::Common::kernMapToBoolean;
        using StreamCompaction::Common::kernScatter;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernPrescan(int n, int* odata, int* idata, int* sum) {
            extern __shared__ int temp[MAX_SHARE_SIZE];
            int idx = threadIdx.x;
            int bid = blockIdx.x;
            int blockOffset = bid * MAX_ELEMENTS_PER_BLOCK;
            int leafNum = MAX_ELEMENTS_PER_BLOCK;

            int offset = 1;
            int ai = idx;
            int bi = idx + (leafNum >> 1);
            int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
            temp[ai + bankOffsetA] = ai + blockOffset < n ? idata[ai + blockOffset] : 0;
            temp[bi + bankOffsetB] = bi + blockOffset < n ? idata[bi + blockOffset] : 0;
            
            for (int d = leafNum >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (idx < d) {
                    int ai = offset * (2 * idx + 1) - 1;
                    int bi = offset * (2 * idx + 2) - 1;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);
                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }

            __syncthreads();
            if (idx == 0) {
                // The last element in a block
                int index = leafNum - 1 + CONFLICT_FREE_OFFSET(leafNum - 1);
                sum[bid] = temp[index];
                temp[index] = 0;
            }

            for (int d = 1; d < leafNum; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (idx < d) {
                    int ai = offset * (2 * idx + 1) - 1;
                    int bi = offset * (2 * idx + 2) - 1;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);
                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();
            if (ai + blockOffset < n)
                odata[ai + blockOffset] = temp[ai + bankOffsetA];

            if (bi + blockOffset < n)
                odata[bi + blockOffset] = temp[bi + bankOffsetB];
        }

        __global__ void kernAdd(int n, int* valus, int* prefix_sum)
        {
            int idx = threadIdx.x;
            int bid = blockIdx.x;
            int block_offset = bid * MAX_ELEMENTS_PER_BLOCK;
            int ai = idx + block_offset;
            int bi = idx + (MAX_ELEMENTS_PER_BLOCK >> 1) + block_offset;

            if (ai < n)
            {
                valus[ai] += prefix_sum[bid];
            }
            if (bi < n)
            {
                valus[bi] += prefix_sum[bid];
            }
        }

        void recursiveScan(int n, int* dev_odata, int* dev_idata) {
            int* dev_sum, *dev_prefix_sum;
            int blockNum = (n + MAX_ELEMENTS_PER_BLOCK - 1) / MAX_ELEMENTS_PER_BLOCK;
            cudaMalloc((void**)&dev_sum, blockNum * sizeof(int));
            checkCUDAError("cudaMalloc dev_sum failed!");
            cudaMalloc((void**)&dev_prefix_sum, blockNum * sizeof(int));
            checkCUDAError("cudaMalloc dev_prefix_sum failed!");

            dim3 fullBlocksPerGrid(blockNum);
            
            kernPrescan << <fullBlocksPerGrid, MAX_THREADS_PER_BLOCK >> > (n, dev_odata, dev_idata, dev_sum);
            checkCUDAError("kernPrescan executed failed!");

            cudaDeviceSynchronize();
            if (blockNum != 1) {
                recursiveScan(blockNum, dev_prefix_sum, dev_sum);
                kernAdd << <fullBlocksPerGrid, MAX_THREADS_PER_BLOCK >> > (n, dev_odata, dev_prefix_sum);
                checkCUDAError("kernAdd executed failed!");
                cudaDeviceSynchronize();
            }

            cudaFree(dev_sum);
            cudaFree(dev_prefix_sum);
            checkCUDAError("cudaFree failed!");
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_odata;
            int* dev_idata;
            int sizeInBytes = n * sizeof(int);
            cudaMalloc((void**)&dev_odata, sizeInBytes);
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, sizeInBytes);
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_idata failed!");
            
            timer().startGpuTimer();

            recursiveScan(n, dev_odata, dev_idata);
            
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            cudaFree(dev_odata);
            cudaFree(dev_idata);
            checkCUDAError("cudaFree failed!");
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int* dev_bool_data, *dev_scan_data;
            int* dev_idata, *dev_odata;
            cudaMalloc((void**)&dev_bool_data, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bool_data failed!");
            cudaMalloc((void**)&dev_scan_data, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_scan_data failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_idata failed!");
            
            timer().startGpuTimer();
            int blockSize = 64;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bool_data, dev_idata);

            recursiveScan(n, dev_scan_data, dev_bool_data);
            
            kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bool_data, dev_scan_data);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");
            int count;
            cudaMemcpy(&count, dev_scan_data + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            count += idata[n - 1] ? 1 : 0;

            cudaFree(dev_bool_data);
            cudaFree(dev_scan_data);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            checkCUDAError("cudaFree failed!");
            return count;
        }
    }
}

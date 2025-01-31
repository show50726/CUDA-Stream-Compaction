#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
           
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }

            // Simulate GPU scan, only effective when n is the power of 2
            /*memcpy(odata, idata, sizeof(int) * n);

            int iteration = ilog2ceil(n);
            int interval = 2;
            for (int i = 0; i < iteration; i++) {
                for (int j = 0; j < n; j += interval) {
                    odata[j + interval - 1] += odata[j + interval / 2 - 1];
                }
                interval *= 2;
            }
  
            odata[n - 1] = 0;
            for (int i = iteration - 1; i >= 0; i--) {
                interval /= 2;
                for (int j = 0; j < n; j += interval) {
                    int t = odata[j + interval/2 - 1];
                    odata[j + interval/2 - 1] = odata[j + interval - 1];
                    odata[j + interval - 1] = t + odata[j + interval - 1];
                }
            }*/

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            int current = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[current++] = idata[i];
                }
            }

            timer().endCpuTimer();
            return current;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int* temp1 = new int[n];
            memset(temp1, 0, sizeof(int) * n);
            int* temp2 = new int[n];
            memset(temp2, 0, sizeof(int) * n);
            for (int i = 0; i < n; i++) {
                temp1[i] = idata[i] > 0 ? 1 : 0;
            }

            // scan
            for (int i = 1; i < n; i++) {
                temp2[i] = temp2[i - 1] + temp1[i - 1];
            }

            int last = 0;
            for (int i = 0; i < n; i++) {
                if (temp1[i] == 1) {
                    last = temp2[i] + 1;
                    odata[temp2[i]] = idata[i];
                }
            }

            timer().endCpuTimer();
            delete[] temp1;
            delete[] temp2;
            return last;
        }
    }
}

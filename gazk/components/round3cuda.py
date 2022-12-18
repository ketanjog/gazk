"""3 Implementations of parallelized round 3 simulations of the PLONK proving system."""

import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import math

class ParallelRound3:
    def __init__(self) -> None:
        self.getSourceModule()

    def getSourceModule(self):
        # write cuda kernel to multiply multiply polynomials
        kernels = r"""

        // naive implementation of polynomial multiplication using atomic add
        __global__ void polymult_naive(const long *A, const long *B, double *sums, uint n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;

            if (i < n && j < n) {
                atomicAdd(sums+i+j, A[i] * B[j]);
            }
        }

        /*  representation of polynomial multiplication in one dimension allowing for much larger
         *  input sizes (up to (2^32-1) * 1024 elements)
         */
        __global__ void polymult_scan(const long *A, const long *B, long *sums, uint n) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;


            if (tid < n*n) {
                int i = tid/n;
                int j = tid%n;
                int tmp;
                if (i+j < n-1) {
                    tmp = j+1;
                }
                else {
                    tmp = n-i;
                }
                int new_index = ((i+j+1) * (i+j))/2 + tmp -1;
                sums[new_index] = A[tid/n] * B[j];

                // printf("i=%d, j=%d, tid=%d, exp:%d\n", l, k, i, ((l+1)*l)/2+k);
                // Debug. No macro check for performance.
                // printf("A=%d i=%d, j=%d, v1=%d, v2=%d\n", A[i/n], B[j], i, j, n-i/n-1, n-j-1);
            }
        }

        __global__ void scan_inefficient(long *g_idata, long *g_odata, long *aux_arr, int n) {
            __shared__ float temp[1024]; // allocated on invocation
            int tidx = threadIdx.x;
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                // copy over data to shared memory
                bool flag = true;
                temp[tidx] = g_idata[i];
                // while stride is less than len of array
                for (unsigned int stride = 1; stride <= n; stride *= 2) {
                    __syncthreads();
                    if (stride > tidx) {
                        flag = false;
                    }
                    float in1;
                    if (flag) {
                        in1 = temp[tidx - stride];
                    }
                    __syncthreads();
                    if (flag){
                        temp[tidx] += in1;
                    }
                }
                __syncthreads();
                g_odata[i] = temp[tidx];

                if(tidx == 1023) {
                    aux_arr[blockIdx.x] = temp[tidx];
                }
                else if (i == n - 1) {
                    aux_arr[blockIdx.x] = temp[tidx];
                }
            }
        }

        __global__ void h_phase3(float *O, float *I, const int n) {
            const unsigned long int x = blockDim.x*blockIdx.x + threadIdx.x;
            if(blockIdx.x > 0 && x < n) {
                O[x] += I[blockIdx.x-1];
            }
        }

        __global__ void evaluate(long *in, long *out, const int v, const int n) {
            const unsigned long int x = blockDim.x*blockIdx.x + threadIdx.x;
            if(x < n) {
                // assume we have a polynomial of the form anxn + an-1xn-1 + ... + a1x + a0
                out[x] = in[x] * pow(v, n-x-1);
            }
        }

        // polynomial multiplication using the fast fourier transform
        __global__ void polymult_fft(const int *A, const int *B, int *sums, double *o, const int v, uint n) {


        }

        """
        self.module_gpu = SourceModule(kernels)
        self.scan_blockdim = (1024, 1, 1)

    def naive(self, A, B):
        # polynomial multiplication of A and B as numpy arrays
        blockDim = (32, 32, 1)
        gridDim = (math.ceil(len(A)/1024), math.ceil(len(B)/1024), 1)

        start = cuda.Event()
        end = cuda.Event()

        func = self.module_gpu.get_function("polymult_naive")

        # create local arrays
        # A = [2, 1, 6]
        # B = [3, 2, 5]

        len_A = len(A)
        len_B = len(B)
        # Must ensure that A and B are of the same size
        if len_A != len_B:
            if len_A > len_B:
                B = np.append(np.zeros(len_A - len_B), B)
            else:
                A = np.append(np.zeros(len_B - len_A), A)

        # Fixme, change 32 bit to 64 bit for larger coefficients
        A = np.array(A, dtype=np.int64)
        B = np.array(B, dtype=np.int64)
        Sums = np.zeros(A.size * B.size, dtype=np.float64)

        # create device arrays
        A_gpu = cuda.mem_alloc(A.nbytes)
        B_gpu = cuda.mem_alloc(B.nbytes)

        # sum size is 2*dim -1
        Sums_gpu = cuda.mem_alloc(Sums.nbytes)

        # copy arrays to device
        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)
        cuda.memcpy_htod(Sums_gpu, Sums)

        cuda.Context.synchronize()
        start.record()
        # func(A_gpu, B_gpu, Sums_gpu, o_gpu, np.int32(1), np.int32(A.size), block=blockDim, grid=gridDim)
        func(A_gpu, B_gpu, Sums_gpu, np.int32(A.size), block=blockDim, grid=gridDim)
        end.record()
        cuda.Context.synchronize()

        cuda.memcpy_dtoh(Sums, Sums_gpu)

        print(Sums)

        return Sums, end.time_since(start)/1000 # return in seconds


    def polymult_with_scan(self, A, B):
        # polynomial multiplication of A and B as numpy arrays
        blockDim = (min(1024, len(A)*len(B)), 1, 1)
        gridDim = (math.ceil(len(A)/1024), 1, 1)

        start = cuda.Event()
        end = cuda.Event()

        func = self.module_gpu.get_function("polymult_scan")
        scan = self.module_gpu.get_function("scan_inefficient")
        h_phase3 = self.module_gpu.get_function("h_phase3")

        len_A = len(A)
        len_B = len(B)
        # Must ensure that A and B are of the same size
        if len_A != len_B:
            if len_A > len_B:
                B = np.append(np.zeros(len_A - len_B), B)
            else:
                A = np.append(np.zeros(len_B - len_A), A)

         # Fixme, change 32 bit to 64 bit for larger coefficients
        A = np.array(A, dtype=np.int64)
        B = np.array(B, dtype=np.int64)
        Sums = np.zeros(A.size * B.size, dtype=np.int64)
        o = np.zeros(1, dtype=np.float64)

        # print(A)
        # print(B)

        total_time = 0

        # create device arrays
        A_gpu = cuda.mem_alloc(A.nbytes)
        B_gpu = cuda.mem_alloc(B.nbytes)
        o_gpu = cuda.mem_alloc(o.nbytes)

        # sum size is 2*dim -1
        Sums_gpu = cuda.mem_alloc(Sums.nbytes)

        # copy arrays to device
        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)
        cuda.memcpy_htod(Sums_gpu, Sums)

        start.record()
        # func(A_gpu, B_gpu, Sums_gpu, o_gpu, np.int32(1), np.int32(A.size), block=blockDim, grid=gridDim)
        func(A_gpu, B_gpu, Sums_gpu, np.int32(A.size), block=blockDim, grid=gridDim)
        end.record()

        cuda.Context.synchronize()
        total_time += end.time_since(start)/1000
        cuda.Context.synchronize()

        cuda.memcpy_dtoh(Sums, Sums_gpu)

        orig_size = len(Sums)

        # how many times to recurse?
        n = int(np.ceil(math.log(len(Sums), 1024)))

        i_gpu = cuda.mem_alloc(Sums.nbytes)
        cuda.memcpy_htod(i_gpu, Sums)

        # result list
        results = []
        # intermediate result list
        intermediate_results = [i_gpu]
        gridsizes = [orig_size]

        for i in range(n):
            o_gpu = cuda.mem_alloc(gridsizes[-1] * np.int64().nbytes)
            results.append(o_gpu)
            gridsizes.append((gridsizes[i] - 1)//self.scan_blockdim[0] + 1)
            aux_arr = cuda.mem_alloc(gridsizes[-1] * np.int64().nbytes)
            intermediate_results.append(aux_arr)


            start.record()
            scan(intermediate_results[i], results[i], intermediate_results[-1], np.int32(gridsizes[i]), block=self.scan_blockdim, grid=(gridsizes[-1], 1, 1))
            end.record()
            cuda.Context.synchronize()
            total_time += end.time_since(start)/1000
            cuda.Context.synchronize()

        for i in range(n-1)[::-1]:
            start.record()
            h_phase3(results[i], results[i+1], np.int32(gridsizes[i]), block=self.scan_blockdim, grid=(gridsizes[i+1], 1, 1))
            end.record()
            cuda.Context.synchronize()
            total_time += end.time_since(start)/1000
            cuda.Context.synchronize()

        o_dev = np.empty_like(Sums, dtype=np.int64)
        cuda.memcpy_dtoh(o_dev, results[0])

        #  extract the final sums
        final_sums = np.empty_like(o_dev, dtype=np.int64)
        for i in range(0, 2*len_A-1):
            final_sums[i] = o_dev[i]

        # print(final_sums)
        return o_dev, total_time

if __name__ == "__main__":
    pr = ParallelRound3()
    f = 10**4
    s = pr.naive([10]*f, [5]*f)
    r = pr.polymult_with_scan([1]*f, [5]*f)
    print(r)
    print(s)
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
        self.grid_dim = None
        self.block_dim = None
        self.getSourceModule()

    def getSourceModule(self):
        # write cuda kernel to multiply multiply polynomials
        naive = r"""

        __global__ void naive(const int *A, const int *B, int *sums, float *o, const int v, uint n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = i % n;
            int res = 0;
            if (i < n*n) {
                res = A[i/n] * B[j] * pow(v, n-i/n-1) * pow(v, n-j-1);
                sums[i] = res;

                // Debug. No macro check for performance.
                //printf("A=%d, B=%d, v=%d, i=%d, j=%d, v1=%d, v2=%d\n", A[i/n], B[j], v, i, j, n-i/n-1, n-j-1);
            }

            __syncthreads();

            atomicAdd(o, res);
        }
        """
        self.module_naive_gpu = SourceModule(naive)

    def naive(self, A, B):
        # polynomial multiplication of A and B as numpy arrays

        # <<<<<<<FOR NOW, WE DO NOT USE A and B!!!!!!>>>>>>
        start = cuda.Event()
        end = cuda.Event()

        func = self.module_naive_gpu.get_function("naive")

        # create local arrays
        A = [2, 1, 6]
        B = [3, 2, 5]

        # Must ensure that A and B are of the same size
        len_A = len(A)
        len_B = len(B)
        if len_A != len_B:
            if len_A > len_B:
                B = np.append(np.zeros(len_A - len_B), B)
            else:
                A = np.append(np.zeros(len_B - len_A), A)

        # Fixme, change 32 bit to 64 bit for larger coefficients
        A = np.array(A, dtype=np.int32)
        B = np.array(B, dtype=np.int32)
        Sums = np.zeros(A.size * B.size, dtype=np.int32)
        o = np.zeros(1, dtype=np.float32)

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
        cuda.memcpy_htod(o_gpu, o)

        func(A_gpu, B_gpu, Sums_gpu, o_gpu, np.int32(2), np.int32(A.size), block=(16, 1, 1), grid=(1, 1, 1))

        cuda.memcpy_dtoh(Sums, Sums_gpu)
        cuda.memcpy_dtoh(o, o_gpu)

        print(Sums)
        print(o)



if __name__ == "__main__":
    pr = ParallelRound3()
    pr.naive(1,2)
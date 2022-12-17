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
        # write cuda kernel to multiply multiple polynomials

        #
        naive = """
        __global__ void naive(const int *A, const int *B, int *sums, const int v, uint n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;

            if (i < n && j < n) {
                sums[i + j] += A[i] * B[j];
            }
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
        Sums = np.zeros(A.size + B.size - 1, dtype=np.int32)

        # create device arrays
        A_gpu = cuda.mem_alloc(A.nbytes)
        B_gpu = cuda.mem_alloc(B.nbytes)

        # sum size is 2*dim -1
        Sums_gpu = cuda.mem_alloc(Sums.nbytes)

        # copy arrays to device
        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)
        cuda.memcpy_htod(Sums_gpu, Sums)

        func(A_gpu, B_gpu, Sums_gpu, np.int32(1), np.int32(A.size), block=(16, 1, 1), grid=(1, 1, 1))

        cuda.memcpy_dtoh(Sums, Sums_gpu)

        print(Sums)



if __name__ == "__main__":
    pr = ParallelRound3()
    pr.naive(1,2)
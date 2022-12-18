"""
This class implements polynomial multiplication using the FFT algorithm with CUDA.
"""

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time
import math

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.fft as cu_fft

import random


class Round3FFT():
    def __init__(self, N, block_dim=1024, grid_dim=1):
        self.N = N
        self.grid_dim = None
        self.block_dim = None
        # self.getSourceModule()

    # def getSourceModule(self):
    #     fft=r"""
    #     #include <cuComplex.h>
    #     #include <cufft.h>
        

    #     __global__ void multiplyWithFFT(cuComplex *a, cuComplex *b, cuComplex *c, cufftHandle fftPlan, int n) {
    #         // Get the thread index
    #         int i = blockIdx.x * blockDim.x + threadIdx.x;
            
    #         // Check if the thread index is within bounds
    #         if (i < n) {
    #             // Perform FFT on the input arrays
    #             cufftComplex fa, fb;
    #             fa.x = a[i].x;
    #             fa.y = a[i].y;
    #             fb.x = b[i].x;
    #             fb.y = b[i].y;
    #             cufftExecC2C(fftPlan, &fa, &fa, CUFFT_FORWARD);
    #             cufftExecC2C(fftPlan, &fb, &fb, CUFFT_FORWARD);
                
    #             // Multiply the FFTs
    #             cuComplex product;
    #             product.x = fa.x * fb.x - fa.y * fb.y;
    #             product.y = fa.x * fb.y + fa.y * fb.x;
                
    #             // Perform inverse FFT to get the element-wise product
    #             cufftExecC2C(fftPlan, &product, &product, CUFFT_INVERSE);
    #             c[i].x = product.x / n;
    #             c[i].y = product.y / n;
    #         }
    #     }

    #     """

    #     self.module_fft_gpu = SourceModule(fft)

    def fft(self, N):
        # Set the polynomials to be multiplied
        
        poly1 = np.array([random.randint(0, 100) for i in range(N)], dtype=np.complex64)
        poly2 = np.array([random.randint(0, 100) for i in range(N)], dtype=np.complex64)
        print("Size of array is: " + str(len(poly1)))

        # Transfer the polynomials to the GPU
        poly1_gpu = gpuarray.to_gpu(poly1)
        poly2_gpu = gpuarray.to_gpu(poly2)

        # Set the FFT plan and execute it
        start = time.time()
        plan = cu_fft.Plan(poly1_gpu.shape, np.complex64, np.complex64)
        cu_fft.fft(poly1_gpu, poly1_gpu, plan)
        cu_fft.fft(poly2_gpu, poly2_gpu, plan)

        # Multiply the transformed polynomials
        poly1_gpu *= poly2_gpu

        # Execute the inverse FFT
        cu_fft.ifft(poly1_gpu, poly1_gpu, plan, True)
        

        # Transfer the result back to the CPU and print it
        result = poly1_gpu.get()
        end = time.time()
        _time = (end - start)/ 1000

        return result, _time




if __name__ == "__main__":
    N = 1000000
    block_dim = 1024
    grid_dim = 1
    start = time.time()
    round3 = Round3FFT(N, block_dim, grid_dim)
    end = time.time()
    
    res, time = round3.fft(N)
    print("Time: ", (end-start)/1000)
    print(res[:20])





            
                

                

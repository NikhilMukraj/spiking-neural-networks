import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule


# should be read from file
mod = SourceModule("""
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){
        out[i] = a[i] + b[i];
    }
}
""")

vector_add = mod.get_function("vector_add")

N = 1000

block_size = 128
grid_size = (N + block_size - 1) // block_size

a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)

dest = np.zeros_like(a)

block_size = 128
grid_size = (N + block_size - 1) // block_size

vector_add(drv.Out(dest), drv.In(a), drv.In(b), np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1))

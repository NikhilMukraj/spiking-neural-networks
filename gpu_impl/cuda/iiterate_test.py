import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule


# should be read from file
mod = SourceModule("""
__global__ void iterate(
    float *voltage,
    float *w,
    float *dv,
    float *dw,
    float *alpha,
    float *beta,
    float *c,
    float *d,
    float *dt,
    float *tau_m,
    float *v_th,
    int *is_spiking
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        step = dt[i] / tau_m[i]
        dv[i] =  (
            0.04 * voltage[i] * voltage[i] + 5. * voltage[i] + 140. - w[i] + input[i]
        ) * step;

        dw[i] =   (
            alpha[i] * (beta[i] * voltage[i] - w[i])
        ) * step;
    }

    for (int i = index; i < n; i += stride) {
        voltage[i] += dv;

        if (voltage[i] >= v_th[i]) {
            is_spiking[i] = 1;
            voltage[i] = c[i];
            w[i] += d[i];
        } else {
            is_spiking[i] = 0;
            w[i] += dw;
        }
    }
}
""")

iterate = mod.get_function('iterate')

N = 100

block_size = 32
grid_size = (N + block_size - 1) // block_size

voltage = np.ones(N, dtype=np.float32) * -60
w = np.ones(N, dtype=np.float32) * 30
dv = np.zeros(N, dtype=np.float32)
dw = np.zeros(N, dtype=np.float32)
alpha = np.ones(N, dtype=np.float32) * 0.01
beta = np.ones(N, dtype=np.float32) * 0.25
c = np.ones(N, dtype=np.float32) * -55
d = np.ones(N, dtype=np.float32) * 8
dt = np.ones(N, dtype=np.float32) * 0.5
tau_m = np.ones(N, dtype=np.float32) * 10
v_th = np.ones(N, dtype=np.float32) * 35
is_spiking = np.zeros(N, dtype=np.int32)

block_size = 128
grid_size = (N + block_size - 1) // block_size

iterate(
    drv.Out(voltage), 
    drv.In(w), 
    drv.In(dv), 
    drv.In(dw), 
    drv.In(alpha), 
    drv.In(beta), 
    drv.In(c), 
    drv.In(d), 
    drv.In(dt), 
    drv.In(tau_m), 
    drv.In(v_th), 
    drv.In(is_spiking), 
    np.int32(N), 
    block=(block_size, 1, 1), 
    grid=(grid_size, 1)
)

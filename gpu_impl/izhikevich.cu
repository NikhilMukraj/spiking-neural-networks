struct IzhikevichNeuron {
    float voltage;
    float w;
    float alpha;
    float beta;
    float c;
    float d;
    float dt;
    float tau_m;
    float v_th;
};

// __global__ void aggregate_inputs(float *voltage, float **adjcanecy_matrix, int n, float *inputs) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;

//     for (int i = index; i < n; i += stride) {
//         for (int j = 0; j < n; j++) {
//             if (adjcanecy_matrix[index][j] != 0) {
//                 // could be modified to translate or scale voltage before input
//                 inputs[index] += voltage[j] 
//             }
//         }
//     }
// }

// try rewrite with struct
__global__ void iterate(
    float *input,
    float *voltage,
    float *w,
    float *dv,
    float *dw,
    float *alpha,
    float *beta,
    float *c,
    float *d,
    float dt,
    float *tau_m,
    float *v_th,
    int *is_spiking,
    int n
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        step = dt / tau_m[i];
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

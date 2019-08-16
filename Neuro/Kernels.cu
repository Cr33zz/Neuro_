__global__ void Elu(int n, float* __restrict input, float* __restrict result, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        result[i] = input[i] > 0 ? input[i] : alpha * (exp(input[i]) - 1);
}

__global__ void EluGrad(int n, float* __restrict output, float* __restrict outputGradient, float* __restrict result, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        result[i] = (output[i] > 0 ? 1 : (output[i] + alpha)) * outputGradient[i];
}
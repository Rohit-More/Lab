#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('nvcc --version')


# In[4]:


get_ipython().system('pip install git+https://github.com/andreinechaev/nvcc4jupyter.git')


# In[5]:


get_ipython().run_line_magic('load_ext', 'nvcc_plugin')


# In[6]:


get_ipython().run_cell_magic('cu', '', '\n#include <stdio.h>\n\n// CUDA kernel for vector addition\n__global__ void vectorAdd(int* a, int* b, int* c, int size) \n{\n    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n    if (tid < size) {\n        c[tid] = a[tid] + b[tid];\n    }\n}\n\nint main() \n{\n    int size = 100;  // Size of the vectors\n    int* a, * b, * c;    // Host vectors\n    int* dev_a, * dev_b, * dev_c;  // Device vectors\n\n    // Allocate memory for host vectors\n    a = (int*)malloc(size * sizeof(int));\n    b = (int*)malloc(size * sizeof(int));\n    c = (int*)malloc(size * sizeof(int));\n\n    // Initialize host vectors\n    for (int i = 0; i < size; i++) {\n        a[i] = i;\n        b[i] = i;\n    }\n\n    // Allocate memory on the device for device vectors\n    cudaMalloc((void**)&dev_a, size * sizeof(int));\n    cudaMalloc((void**)&dev_b, size * sizeof(int));\n    cudaMalloc((void**)&dev_c, size * sizeof(int));\n\n    // Copy host vectors to device\n    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);\n    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);\n\n    // Launch kernel for vector addition\n    int blockSize = 256; //threads\n    int gridSize = (size + blockSize - 1) / blockSize;  //blocks\n    vectorAdd<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, size);\n\n    // Copy result from device to host\n    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);\n\n    // Print result\n    for (int i = 0; i < size; i++) {\n        printf("%d + %d = %d\\n", a[i], b[i], c[i]);\n    }\n\n    // Free device memory\n    cudaFree(dev_a);\n    cudaFree(dev_b);\n    cudaFree(dev_c);\n\n    // Free host memory\n    free(a);\n    free(b);\n    free(c);\n\n    return 0;\n}')


# In[20]:


get_ipython().run_cell_magic('cu', '', '#include <stdio.h>\n\n__global__ void matrixMultiply(int *A, int *B, int *C, int N) {\n  int row = blockIdx.y * blockDim.y + threadIdx.y;\n  int col = blockIdx.x * blockDim.x + threadIdx.x;\n\n  if (row < N && col < N) {\n    int sum = 0;\n    for (int k = 0; k < N; ++k) {\n      sum += A[row * N + k] * B[k * N + col];\n    }\n    C[row * N + col] = sum;\n  }\n}\n\nint main() {\n  int N = 2;\n  int size = N * N * sizeof(int);\n  int *A, *B, *C;\n  int *dev_A, *dev_B, *dev_C;\n\n  // Allocate memory for matrices A, B, and C on the host\n  A = (int *)malloc(size);\n  B = (int *)malloc(size);\n  C = (int *)malloc(size);\n\n  // Initialize matrices A and B\nfor (int i = 0; i < N; ++i) {\n  for (int j = 0; j < N; ++j) {\n    A[i * N + j] = i + j;\n    B[i * N + j] = i * N + j;  \n  }\n}\n\n  printf("initial matrix A:\\n");\n  for (int i = 0; i < N; ++i) {\n    for (int j = 0; j < N; ++j) {\n      printf("%d ", A[i * N + j]);\n    }\n    printf("\\n");\n  }\n\n  printf("initial matrix B:\\n");\n  for (int i = 0; i < N; ++i) {\n    for (int j = 0; j < N; ++j) {\n      printf("%d ", B[i * N + j]);\n    }\n    printf("\\n");\n  }\n\n  // Allocate memory for matrices A, B, and C on the device\n  cudaMalloc(&dev_A, size);\n  cudaMalloc(&dev_B, size);\n  cudaMalloc(&dev_C, size);\n\n  // Copy matrices A and B from host to device\n  cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);\n  cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);\n\n  // Define grid and block dimensions\n  dim3 dimBlock(16, 16);\n  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);\n\n  // Launch the matrix multiplication kernel\n  matrixMultiply<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);\n\n  // Copy the result matrix C from device to host\n  cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);\n\n  // Print the result matrix\n  printf("Result matrix C:\\n");\n  for (int i = 0; i < N; ++i) {\n    for (int j = 0; j < N; ++j) {\n      printf("%d ", C[i * N + j]);\n    }\n    printf("\\n");\n  }\n\n  // Free device memory\n  cudaFree(dev_A);\n  cudaFree(dev_B);\n  cudaFree(dev_C);\n\n  // Free host memory\n  free(A);\n  free(B);\n  free(C);\n\n  return 0;\n}')


# In[25]:


# MATRIX MULTIPLICATION

%%cu

#include <stdio.h>

(/, CUDA, kernel, for, matrix, multiplication)
__global__ void matrixMul(int* a, int* b, int* c, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (row < rowsA && col < colsB) {
        for (int i = 0; i < colsA; i++) {
            sum += a[row * colsA + i] * b[i * colsB + col];
        }
        c[row * colsB + col] = sum;
    }
}

int main() {
    int rowsA = 2;  // Rows of matrix A
    int colsA = 2;  // Columns of matrix A
    int rowsB = colsA; // Rows of matrix B
    int colsB = 2;  // Columns of matrix B

    int* a, * b, * c;  // Host matrices
    int* dev_a, * dev_b, * dev_c;  // Device matrices

    // Allocate memory for host matrices
    a = (int*)malloc(rowsA * colsA * sizeof(int));
    b = (int*)malloc(rowsB * colsB * sizeof(int));
    c = (int*)malloc(rowsA * colsB * sizeof(int));

    // Initialize host matrices
    for (int i = 0; i < rowsA * colsA; i++) {
        a[i] = i;
    }
    for (int i = 0; i < rowsB * colsB; i++) {
        b[i] = 2 * i;
    }

    // Allocate memory on the device for device matrices
    cudaMalloc((void**)&dev_a, rowsA * colsA * sizeof(int));
    cudaMalloc((void**)&dev_b, rowsB * colsB * sizeof(int));
    cudaMalloc((void**)&dev_c, rowsA * colsB * sizeof(int));

    // Copy host matrices to device
    cudaMemcpy(dev_a, a, rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, rowsB * colsB * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x, (rowsA + blockSize.y - 1) / blockSize.y);

    // Launch kernel for matrix multiplication
    matrixMul<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, rowsA, colsA, colsB);

    // Copy result from device to host
    cudaMemcpy(c, dev_c, rowsA * colsB * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Result:\n");
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            printf("%d ", c[i * colsB + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // Free host memory
    free(a);
    free(b);
    free(c);

    return 0;
}


# In[ ]:





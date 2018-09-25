#include <stdio.h>

void init(double *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i%3;
  }
}

__global__
void doubleElements(double *a, int N)
{

  /*
   * Use a grid-stride loop so each thread does work
   * on more than one element in the array.
   */

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] *= 2;
  }
}



int main()
{
  int N = 10000;
  double *a;

  size_t size = N * sizeof(double);
  cudaMallocManaged(&a, size);

  init(a, N);

  size_t threads_per_block = 256;
  size_t number_of_blocks = 32;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  cudaDeviceSynchronize();

printf("\nresult: ");
  for ( int i = 0; i < N ; i ++ )
{
printf("%lf ", a[i]);
}

  cudaFree(a);
}


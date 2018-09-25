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
void doubleElements(double *a, int N, int color)
{

  /*
   * Use a grid-stride loop so each thread does work
   * on more than one element in the array.
   */

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  if ( color == 1 ) 
  {

  for (int i = idx; i < N; i += stride)
  {
    a[i] *= 1;
  }

  }

    if ( color == 2 ) 
  {

  for (int i = idx; i < N; i += stride)
  {
    a[i] *= 2;
  }

  } 
      if ( color == 3 ) 
  {

  for (int i = idx; i < N; i += stride)
  {
    a[i] *= 3;
  }

  }


  
}



int main()
{
  int N = 10;
  double *red;

  size_t size = N * sizeof(double);
  cudaMallocManaged(&red, size);
  init(red, N);
  double *green;
  cudaMallocManaged(&green, size);
  init(green, N);
  double *blue;
  cudaMallocManaged(&blue, size);
  init(blue, N);


  size_t threads_per_block = 256;
  size_t number_of_blocks = 32;

  doubleElements<<<number_of_blocks, threads_per_block>>>(red, N, 1);
  doubleElements<<<number_of_blocks, threads_per_block>>>(blue, N, 2);
  doubleElements<<<number_of_blocks, threads_per_block>>>(green, N, 3);
  cudaDeviceSynchronize();

printf("\nresult: ");
  for ( int i = 0; i < N ; i ++ )
{
printf("%lf ", red[i]);
}
  for ( int i = 0; i < N ; i ++ )
{
printf("%lf ", green[i]);
}

  for ( int i = 0; i < N ; i ++ )
{
printf("%lf ", blue[i]);
}

  cudaFree(red);
  cudaFree(green);
  cudaFree(blue);
}


#include <stdio.h>

void init(double *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i%3;
  }
}

struct position {
     int x;
     int y;
};

/// convert a 2D position to a 1D index
/// assumes bottom left corner of image is 0,0 and index 1


long get1dIndex( int width,  int x, int y) {

    return y * width + x;
}

/// inverse of 2D to 1D mapping function
/// sends back x,y values in tuple from index
void get_Position( int width, int id, struct position *pos) {
     int xx = 0;
     int yy = 0;
    // struct position pos;
    xx = id / width;
    yy = id % width;
    pos->x = yy;
    pos->y = xx;
    // return pos;
}

__global__
void doubleElements(double *a, int N, int color )
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
    int x = i/4; 
    int y = i%4;
    a[i] = a[i] + 2*x + y*i;

  }

  }

    if ( color == 2 ) 
  {

  for (int i = idx; i < N; i += stride)
  {
    int x = i/4; 
    int y = i%4;
    a[i] = a[i] + 2*x + y*i;
  }

  } 
      if ( color == 3 ) 
  {

  for (int i = idx; i < N; i += stride)
  {
    int x = i/4; 
    int y = i%4;
    a[i] = a[i] + 2*x + y*i;
  }

  }
}

int main()
{
  int N = 12;
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



#include <stdio.h>
#include <stdio.h>
extern "C" {
#include "bmpfile.h"
}

/*Mandelbrot values*/
#define RESOLUTION 8700.0
#define XCENTER -0.55
#define YCENTER 0.6
#define MAX_ITER 1000
#define WIDTH 50.0
#define HEIGHT 80.0

/*Colour Values*/
#define COLOUR_DEPTH 255
#define COLOUR_MAX 240.0
#define GRADIENT_COLOUR_MAX 230.0
#define FILENAME "my_mandelbrot_fractal.bmp"

void init(double *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = 0;
  }
}

__global__
void doubleElements(double *ar, int N, int color )
{

  /*
   * Use a grid-stride loop so each thread does work
   * on more than one element in the array.
   */

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

    int xoffset = -(WIDTH - 1) /2;
    int yoffset = (HEIGHT -1) / 2;

  for (int i = idx; i < N; i += stride)
  {

   //// processing function

        int col = i/4;
        int row = i%4;

        //Determine where in the mandelbrot set, the pixel is referencing
        double x = XCENTER + (xoffset + col) / RESOLUTION;
        double y = YCENTER + (yoffset - row) / RESOLUTION;

        //Mandelbrot stuff

        double a = 0;
        double b = 0;
        double aold = 0;
        double bold = 0;
        double zmagsqr = 0;
        int iter =0;
	    double x_col;
        //Check if the x,y coord are part of the mendelbrot set - refer to the algorithm
        while(iter < MAX_ITER && zmagsqr <= 4.0){
           ++iter;
	    a = (aold * aold) - (bold * bold) + x;
        b = 2.0 * aold*bold + y;
           zmagsqr = a*a + b*b;
           aold = a;
           bold = b;

        }

        /* Generate the colour of the pixel from the iter value */
        /* You can mess around with the colour settings to use different gradients */
        /* Colour currently maps from royal blue to red */

        x_col =  (COLOUR_MAX - (( ((float) iter / ((float) MAX_ITER) * GRADIENT_COLOUR_MAX))));

        double posSlope = (COLOUR_DEPTH- 1)/60;
        double negSlope = (1-COLOUR_DEPTH)/60;
          if ( color == 0)
          {

                if( x_col < 60 )
                {
                    ar[i] = COLOUR_DEPTH;
                }
                if (  x_col >= 60 && x_col < 120 )
                {
                    ar[i] = negSlope*x+2.0*COLOUR_DEPTH+1;
                }
                if (  x_col >=120 && x_col < 180  )
                {
                    ar[i] = 1;
                }
                if ( x_col >=180  && x_col < 240  )
                {
                    ar[i] = 1;
                }
                if (  x_col>= 240 && x_col < 300  )
                {
                    ar[i] = posSlope*x-4.0*COLOUR_DEPTH+1;
                }
                else
                {
                  ar[i] = COLOUR_DEPTH;
                }

           }
           if ( color == 1 )
                {

                    if ( x_col < 60 )
                    {
                        ar[i] = posSlope*x+1;
                    }
                    if (  x_col >= 60 && x_col < 120 )
                    {
                        ar[i] = COLOUR_DEPTH;
                    }
                    if (  x_col >=120 && x_col < 180  )
                    {
                        ar[i] = COLOUR_DEPTH;
                    }
                    if ( x_col >=180  && x_col < 240  )
                    {
                        ar[i] = negSlope*x+4.0*COLOUR_DEPTH+1;
                    }
                    if (  x_col>= 240 && x_col < 300  )
                    {
                        ar[i] = 1;
                    }
                    else
                    {
                        ar[i] = 1;
                    }
                }
                  if ( color == 2)
                  {

                    if ( x_col < 60 )
                    {
                        ar[2] = 1;
                    }
                    if (  x_col >= 60 && x_col < 120 )
                    {
                        ar[2] = 1;
                    }
                    if (  x_col >=120 && x_col < 180  )
                    {
                        ar[2] = posSlope*x-2.0*COLOUR_DEPTH+1;
                    }

                    if (x_col >=180  && x_col < 240  )
                    {
                        ar[2] = COLOUR_DEPTH;
                    }
                    if (  x_col>= 240 && x_col < 300  )
                    {
                        ar[2] = COLOUR_DEPTH;
                    }
                    else
                    {
                        ar[2] = negSlope*x+6*COLOUR_DEPTH;
                    }
                }
  }

}

int main()
{

  bmpfile_t *bmp;
  rgb_pixel_t pixel = {0, 0, 0, 0};

  bmp = bmp_create(WIDTH, HEIGHT, 32);
  int N = WIDTH*HEIGHT;
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

  doubleElements<<<number_of_blocks, threads_per_block>>>(red, N, 0);
  doubleElements<<<number_of_blocks, threads_per_block>>>(blue, N, 1);
  doubleElements<<<number_of_blocks, threads_per_block>>>(green, N, 2);
  cudaDeviceSynchronize();

// now we have red blue green in 3 arrays and hopefully index of arrays is based on x,y values

     for(int xx = 0; xx < WIDTH; xx++)
     {
        for(int yy = 0; yy < HEIGHT; yy++)
        {
        // i is get i from x , y
        int ii = yy * WIDTH + xx;
        pixel.red = red[ii];
        pixel.green = blue[ii];
	pixel.blue = green[ii];
        bmp_set_pixel(bmp, xx, yy, pixel);
        }
     }

  bmp_save(bmp, FILENAME);
  bmp_destroy(bmp);

  cudaFree(red);
  cudaFree(green);
  cudaFree(blue);
}


#include <stdio.h>

#include <stdio.h>
#include "bmpfile.h"

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
void doubleElements(double *a, int N, int color )
{

  /*
   * Use a grid-stride loop so each thread does work
   * on more than one element in the array.
   */

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;


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
        double color[3];
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
        GroundColorMix(color, x_col, 1, COLOUR_DEPTH);
        double posSlope = (max-min)/60;
        double negSlope = (min-max)/60;

          if ( color = 0)
          {

                if( x < 60 )
                {
                    a[i] = max;
                }
                if ( if x >+ 60 && x < 120 )
                {
                    a[i] = negSlope*x+2.0*max+min;
                }
                if ( if x >=120 && x < 180  )
                {
                    a[i] = min;
                }
                if (if x >=180  && x < 240  )
                {
                    a[i] = min;
                }
                if ( if x>= 240 && x < 300  )
                {
                    a[i] = posSlope*x-4.0*max+min;
                }
                elseif (240 <= x < 300  )
                {
                  a[i] = max;
                }

           }

           if ( color = 1)
                {

                    if( x < 60 )
                    {
                        a[i] = posSlope*x+min;
                    }
                    if ( if x >+ 60 && x < 120 )
                    {
                        a[i] = max;
                    }
                    if ( if x >=120 && x < 180  )
                    {
                        a[i] = max;
                    }
                    if (if x >=180  && x < 240  )
                    {
                        a[i] = negSlope*x+4.0*max+min;
                    }
                    if ( if x>= 240 && x < 300  )
                    {
                        a[i] = min;
                    }
                    else
                    {
                        a[i] = min;
                    }
                }



                  if ( color = 2)
                  {

                    if( x < 60 )
                    {
                        a[2] = min;
                    }
                    if ( if x >+ 60 && x < 120 )
                    {
                        a[2] = min;
                    }
                    if ( if x >=120 && x < 180  )
                    {
                        a[2] = posSlope*x-2.0*max+min;
                    }

                    if (if x >=180  && x < 240  )
                    {
                        a[2] = max;
                    }
                    if ( if x>= 240 && x < 300  )
                    {
                        a[2] = max;
                    }
                    else
                    {
                        a[2] = negSlope*x+6*max;
                    }
                }


  }

}

int main()
{

  bmpfile_t *bmp;
  rgb_pixel_t pixel = {0, 0, 0, 0};
  int xoffset = -(WIDTH - 1) /2;
  int yoffset = (HEIGHT -1) / 2;
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

  doubleElements<<<number_of_blocks, threads_per_block>>>(red, N, 1);
  doubleElements<<<number_of_blocks, threads_per_block>>>(blue, N, 2);
  doubleElements<<<number_of_blocks, threads_per_block>>>(green, N, 3);
  cudaDeviceSynchronize();

// now we have red blue green in 3 arrays and hopefully index of arrays is based on x,y values

     for(xx = 0; xx < WIDTH; xx++)
     {
        for(yy = 0; yy < HEIGHT; yy++)
        {
        // i is get i from x , y
        int ii = yy * WIDTH + xx;
        pixel.red = red[i];
        pixel.green = blue[i];
	    pixel.blue = green[i];
        bmp_set_pixel(bmp, xx, yy, pixel);
        }
     }

  bmp_save(bmp, FILENAME);
  bmp_destroy(bmp);


  cudaFree(red);
  cudaFree(green);
  cudaFree(blue);
}


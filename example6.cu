//
// Created by sean on 9/26/18.
// Program which generates a mandelbrot fractal image using CUDA 
// to compile make clean then make example
// to run ./example 
// enter inputs from terminal and image will be in the directory where program was called from 


#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
#include "bmpfile.h"
}

/*Mandelbrot values*/
#define RESOLUTION 8700.0
#define XCENTER -0.55
#define YCENTER 0.6
#define MAX_ITER 5000

/*Colour Values*/
#define COLOUR_DEPTH 255
#define COLOUR_MAX 240.0
#define GRADIENT_COLOUR_MAX 230.0
#define FILENAME "my_mandelbrot_fractal.bmp"

// Define this to turn on error checking
#define CUDA_ERROR_CHECK
#define CudaSafeCall(err) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

// check a CUDA call for errors 
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif

    return;
}
// call this after executing a kernel etc to make sure everything worked properly 
inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif

    return;
}


// set values of an array to zero
void init(double *a, int N) {
    int i;
    for (i = 0; i < N; ++i) {
        a[i] = 0;
    }
}

// function called on each thread .
// I attempted to prcoess r g b values seperately
// *ar is a double array passed in for each color
// N is the total number of pixels
// color determines whether red, green or blue are being processed
// width and height are the height of the image
__global__
void mandelbrot(double *ar, int N, int color, int WIDTH, int HEIGHT) {

    /*
     * Use a grid-stride loop so each thread does work
     * on more than one element in the array.
     */

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int xoffset = -(WIDTH - 1) / 2;
    int yoffset = (HEIGHT - 1) / 2;


    for (int i = idx; i < N; i += stride) {

        // processing function

        // convert the 1D index into an x ,y position
        int col = i / WIDTH;
        int row = i % WIDTH;

        //Determine where in the mandelbrot set, the pixel is referencing
        double x = XCENTER + (xoffset + col) / RESOLUTION;
        double y = YCENTER + (yoffset - row) / RESOLUTION;

        //Mandelbrot stuff

        double a = 0;
        double b = 0;
        double aold = 0;
        double bold = 0;
        double zmagsqr = 0;
        int iter = 0;
        double x_col;
        //Check if the x,y coord are part of the mendelbrot set - refer to the algorithm
        while (iter < MAX_ITER && zmagsqr <= 4.0) {
            ++iter;
            a = (aold * aold) - (bold * bold) + x;
            b = 2.0 * aold * bold + y;

            zmagsqr = a * a + b * b;

            aold = a;
            bold = b;

        }

        /* Generate the colour of the pixel from the iter value */
        /* You can mess around with the colour settings to use different gradients */
        /* Colour currently maps from royal blue to red */

        x_col = (COLOUR_MAX - ((((float) iter / ((float) MAX_ITER) * GRADIENT_COLOUR_MAX))));

        double posSlope = (COLOUR_DEPTH - 1) / 60;
        double negSlope = (1 - COLOUR_DEPTH) / 60;

        // process the red values if we are processing the red values

        if (color == 0) {

            if (x_col < 60) {
                ar[i] = COLOUR_DEPTH;
            } else if (x_col >= 60 && x_col < 120) {
                ar[i] = negSlope * x_col + 2.0 * COLOUR_DEPTH + 1;
            } else if (x_col >= 120 && x_col < 180) {
                ar[i] = 1;
            } else if (x_col >= 180 && x_col < 240) {
                ar[i] = 1;
            } else if (x_col >= 240 && x_col < 300) {
                ar[i] = posSlope * x_col - 4.0 * COLOUR_DEPTH + 1;
            } else {
                ar[i] = COLOUR_DEPTH;
            }

        }

        // process the green values if this is green

        if (color == 1) {

            if (x_col < 60) {
                ar[i] = posSlope * x + 1;
            } else if (x_col >= 60 && x_col < 120) {
                ar[i] = COLOUR_DEPTH;
            } else if (x_col >= 120 && x_col < 180) {
                ar[i] = COLOUR_DEPTH;
            } else if (x_col >= 180 && x_col < 240) {
                ar[i] = negSlope * x_col + 4.0 * COLOUR_DEPTH + 1;
            } else if (x_col >= 240 && x_col < 300) {
                ar[i] = 1;
            } else {
                ar[i] = 1;
            }
        }


         // process the blue values if this is blue
        if (color == 2) {

            if (x_col < 60) {
                ar[i] = 1;
            } else if (x_col >= 60 && x_col < 120) {
                ar[i] = 1;
            } else if (x_col >= 120 && x_col < 180) {
                ar[i] = posSlope * x_col - 2.0 * COLOUR_DEPTH + 1;
            } else if (x_col >= 180 && x_col < 240) {
                ar[i] = COLOUR_DEPTH;
            } else if (x_col >= 240 && x_col < 300) {
                ar[i] = COLOUR_DEPTH;
            } else {
                ar[i] = negSlope * x_col + 6 * COLOUR_DEPTH;
            }
        }


    }

}

int main() {

    bmpfile_t *bmp; // file to store image
    rgb_pixel_t pixel = {0, 0, 0, 0};


    int WIDTH;
    int HEIGHT;

    // get the user to enter height and width // max is 10000
    // if incorrect values are given program terminates
    printf("\n********Sean's Fractal creator*****");
    printf("\n \n \n");
    printf("\nEnter image width: \n"); // prompt
    scanf("%d", &WIDTH); // read an integer
    printf("\nEnter image height\n"); // prompt
    scanf("%d", &HEIGHT); // read an integer

    if (WIDTH < 0 || WIDTH > 10000) {
        printf("\nPlease enter a valid integer between zero and 10000 for height and width");
        printf("\nProgram will exit");
        exit(0);
    }

    if (WIDTH < 0 || HEIGHT > 10000) {
        printf("\nPlease enter a valid integer between zero and 10000 for height and width");
        printf("\nProgram will exit");
        exit(0);
    }
    printf("\n*****PROCESSING ! Image name is : my_mandelbrot_fractal.bmp ");

    // create a bmp file
    bmp = bmp_create(WIDTH, HEIGHT, 32);
    int N = WIDTH * HEIGHT;

    // set up and allocate memory for 3 arrays to store r g b values for each pixel
    size_t size = N * sizeof(double);
    double *red;
    CudaSafeCall(cudaMallocManaged(&red, size));
    init(red, N);
    double *blue;
    CudaSafeCall(cudaMallocManaged(&blue, size));
    init(blue, N);
    double *green;
    CudaSafeCall(cudaMallocManaged(&green, size));
    init(green, N);


    size_t threads_per_block = 256;
    size_t number_of_blocks = 64;

    // call the function 3 times once for each color in a grid stride formate

    mandelbrot<<< number_of_blocks, threads_per_block >>> (red, N, 0, WIDTH, HEIGHT);
    CudaCheckError();
    mandelbrot<<< number_of_blocks, threads_per_block >>> (blue, N, 1, WIDTH, HEIGHT);
    CudaCheckError();
    mandelbrot<<< number_of_blocks, threads_per_block >>> (green, N, 2, WIDTH, HEIGHT);
    CudaCheckError();
    cudaDeviceSynchronize();

// now we have red blue green in 3 arrays and hopefully index of arrays is based on x,y values
// copy the processed r , g , b values back into the array
    for (int xx = 0; xx < WIDTH; xx++) {
        for (int yy = 0; yy < HEIGHT; yy++) {
            // i is get i from x , y
            int ii = yy * WIDTH + xx;
            pixel.red = red[ii];
            //printf("r %lf ", red[ii]);
            pixel.blue = blue[ii];
            //printf("b %lf ", blue[ii]);
            pixel.green = green[ii];
            //printf("g %lf ", green[ii]);
            bmp_set_pixel(bmp, xx, yy, pixel);
        }
    }

    bmp_save(bmp, FILENAME);
    bmp_destroy(bmp);


    cudaFree(red);
    cudaFree(blue);
    cudaFree(green);
    return(0);
}

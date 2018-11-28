
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int nx, const int ny, float *  image, float *  tmp_image);
void init_image(const int nx, const int ny, float *  image, float *  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
void startProcess(const int local_ncols, const int local_nrows, float *sendbuf, float *recvbuf, const int up, const int down, int tag, float *local_gridcurrent, float *local_gridnext, MPI_Status status);
double wtime(void);

int main(int argc, char *argv[]) {
  int ii, jj, kk;
  int rank,size,flag =0,strlen;
  int tag =0;
  int up, down;
  MPI_Status status;
  int local_nrows, local_ncols, remote_ncols;
  float *local_gridcurrent, *local_gridnext, *sendbuf, *recvbuf, *printbuf;

  enum bool {FALSE,TRUE};
  char hostname[MPI_MAX_PROCESSOR_NAME];


  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the image
  float *image = malloc(sizeof(float)*nx*ny);
  float *tmp_image = malloc(sizeof(float)*nx*ny);

  // Set the input image
  init_image(nx, ny, image, tmp_image);

  MPI_Init(&argc, &argv);
  MPI_Initialized(&flag);
  if(flag!=TRUE){
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }
  MPI_Get_processor_name(hostname,&strlen);

  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // left and right notes
  up = (rank+size-1)%size;
  down = (rank+1) % size;

  //determine local grid size
  //if rows cannot be divided evenly amongst the nodes
  //then just add an extra row to each node that is less than the remainder
  local_nrows = nx/size;
  if(nx%size != 0) {
    if(rank<nx%size)
      local_nrows += 1;
  }
  local_ncols = ny;

  //allocate space for massage buffers, local grid with extra rows
  local_gridcurrent = (float *)malloc(sizeof(float*) * (local_nrows+2) * local_ncols);
  local_gridnext = (float *)malloc(sizeof(float*) * (local_nrows+2) * local_ncols);
  sendbuf = (float*)malloc(sizeof(float)*local_ncols);
  recvbuf = (float*)malloc(sizeof(float)*local_ncols);

  //initialize local gird
  for(ii=1;ii<local_nrows-1;ii++){
    for(jj=0;jj<local_ncols;jj++){
      local_gridcurrent[jj+ii*local_ncols] = image[jj+ii*local_ncols];
    }
  }




  // Call the stencil kernel
  double tic = wtime();

  for (int t = 0; t < niters; ++t) {
    startProcess(local_ncols, local_nrows, sendbuf, recvbuf, up, down, tag, local_gridcurrent, local_gridnext, status);
    startProcess(local_ncols, local_nrows, sendbuf, recvbuf, up, down, tag, local_gridnext, local_gridcurrent, status);
  }
  printf("Hello, world; from host %s: process %d of %d\n", hostname, rank, size);

  double toc = wtime( );
  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");

  MPI_Finalize();

  output_image(OUTPUT_FILE, nx, ny, image);

  free(image);

  return EXIT_SUCCESS;
}

void startProcess(const int local_ncols, const int local_nrows, float *sendbuf, float *recvbuf, const int up, const int down, int tag, float *local_gridcurrent, float *local_gridnext, MPI_Status status){
  //Exchange local grid halo rows
  //first send up and receive down
  for(int jj=0;jj<local_ncols;jj++)
    sendbuf[jj] = local_gridcurrent[jj+local_ncols];

  MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, up, tag,
               recvbuf, local_ncols, MPI_FLOAT, down, tag,
               MPI_COMM_WORLD, &status);

  for(int jj=0;jj<local_ncols;jj++)
    local_gridcurrent[jj+(local_nrows-2)*local_ncols] = recvbuf[jj];

  //now send right receive left
  for(int jj=0;jj<local_ncols;jj++)
    sendbuf[jj] = local_gridcurrent[jj+(local_nrows-2)*local_ncols];

  MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, down, tag,
               recvbuf, local_ncols, MPI_FLOAT, up, tag,
               MPI_COMM_WORLD, &status);

  for(int jj=0;jj<local_ncols;jj++)
    local_gridcurrent[jj] = recvbuf[jj];

  stencil(local_nrows, local_ncols, local_gridcurrent, local_gridnext);
}

void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
  //#pragma GCC ivdep
  //left-up corner
  tmp_image[0] = image[0] * 0.6 +
                      (image[1] +
                      image[ny-1])*0.1;
  //right-up corner
  tmp_image[ny-1] = image[ny-1] * 0.6 +
                      (image[ny-2] +
                      image[ny-1+nx])*0.1;
  //left-down corner
  tmp_image[(nx-1)*ny] = image[(nx-1)*ny] * 0.6 +
                      (image[(nx-1)*ny+1] +
                      image[(nx-2)*ny])*0.1;
  //right-down corner
  tmp_image[nx*ny-1] = image[nx*ny] * 0.6 +
                      (image[nx*ny-1] +
                      image[(nx-1)*ny])*0.1;
  //up and down edges can be in this loop only because nx=ny
  for (int i = 1; i < nx-1; i++) { //i == rows
    //up-edge
    tmp_image[i] = image[i] * 0.6 +
                        (image[i+ny] +
                        image[i+1] +
                        image[i-1])*0.1;
    //left-edge
    tmp_image[i*ny] = image[i*ny] * 0.6 +
                        (image[(i-1)*ny] +
                        image[(i+1)*ny] +
                        image[i*ny+1])*0.1;
    for (int j = 1; j < ny-1; j++) { //j == columns
      //everything in the middle
      tmp_image[j+i*ny] = image[j+i*ny] * 0.6 +
                          (image[j  +(i-1)*ny] +
                          image[j  +(i+1)*ny] +
                          image[j-1+i*ny] +
                          image[j+1+i*ny])*0.1;
    }
    //down-edge
    tmp_image[i+(nx-1)*(ny)] = image[i+(nx-1)*(ny)] * 0.6 +
                        (image[i+(nx-2)*(ny)] +
                        image[i+1+(nx-1)*(ny)] +
                        image[i-1+(nx-1)*(ny)])*0.1;
    //right-edge
    tmp_image[(ny-1)+i*ny] = image[(ny-1)+i*ny] * 0.6 +
                        (image[(ny-1)+(i-1)*ny] +
                        image[(ny-1)+(i+1)*ny] +
                        image[(ny-2)+i*ny])*0.1;
  }
}

// Create the input image
void init_image(const int nx, const int ny, float *  image, float *  tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[j+i*ny] = 0.0;
    }
  }

  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[jj+ii*ny] = 100.0;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, float *image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  float maximum = 0.0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}


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
void startProcess(const int local_ncols, const int local_nrows, float *sendbuf, float *recvbuf, const int up, const int down, const int rank, const int size, int tag, float *local_gridcurrent, float *local_gridnext, MPI_Status status);
double wtime(void);

int main(int argc, char *argv[]) {
  enum bool {FALSE,TRUE};
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int rank,size,flag =0,strlen;
  MPI_Init(&argc, &argv);
  MPI_Initialized(&flag);
  if(flag!=TRUE){
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }
  MPI_Get_processor_name(hostname,&strlen);

  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  int ii, jj, kk;
  int tag =0;
  int up, down;
  MPI_Status status;
  int local_nrows, local_ncols, remote_ncols;
  float *local_gridcurrent, *local_gridnext, *sendbuf, *recvbuf, *printbuf;




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
  init_image(nx, ny, image, tmp_image);

  // left and right notes
  up = (rank+size-1)%size;
  down = (rank+1) % size;

  //determine local grid size
  //if rows cannot be divided evenly amongst the nodes
  //then just add an extra row to each node that is less than the remainder
  local_nrows = (ny/size) +2;
  if(ny%size != 0) {
    if(rank<ny%size)
      local_nrows += 1;
  }
  local_ncols = nx;

  //allocate space for message buffers, local grid with extra rows
  local_gridcurrent = (float*)malloc(sizeof(float) * (local_nrows+2) * local_ncols);
  local_gridnext = (float *)malloc(sizeof(float) * (local_nrows+2) * local_ncols);
  sendbuf = (float*)malloc(sizeof(float)*local_ncols);
  recvbuf = (float*)malloc(sizeof(float)*local_ncols);
  // gatherbuf = (float*)malloc(sizeof(float)*local_ncols);

  //initialize local girds
  for(ii=1;ii<local_nrows-1;ii++){
    for(jj=0;jj<local_ncols;jj++){
      int remainder = ny%size;
      if(rank<remainder || remainder==0){
        local_gridcurrent[jj + ii*local_ncols] = image[jj + (ii-1)*local_ncols + rank*(local_nrows-2)*local_ncols];
        local_gridnext[jj + ii*local_ncols] = 0.0;
      }else{
        local_gridcurrent[jj + ii*local_ncols] = image[jj + (ii-1)*local_ncols + (remainder)*(local_nrows-1)*local_ncols + (rank-(remainder))*(local_nrows-2)*local_ncols];
        local_gridnext[jj + ii*local_ncols] = 0.0;
      }
    }
  }
  if(rank == MASTER){
    for(int jj =0; jj<local_ncols;jj++){
      local_gridcurrent[jj] = 0.0;
    }
  } else if(rank == (size-1)){
    for(int jj =0; jj<local_ncols;jj++){
      local_gridcurrent[jj+(local_nrows-1)*local_ncols] = 0.0;
    }
  }




  // Call the stencil kernel
  double tic = wtime();

  for (int t = 0; t < niters; ++t) {
    startProcess(local_ncols, local_nrows, sendbuf, recvbuf, up, down, rank,size, tag, local_gridcurrent, local_gridnext, status);
    startProcess(local_ncols, local_nrows, sendbuf, recvbuf, up, down, rank,size, tag, local_gridnext, local_gridcurrent, status);
  }


  double toc = wtime( );

  // After finished, start sending everything back to the master node to finish up the process
  // and gather the original image.
  if(rank == MASTER){
    for(int i=1; i<local_nrows-1; i++){

      for(int jj=0;jj<local_ncols;jj++){
        image[jj + (i-1)*local_ncols] = local_gridcurrent[jj + i*local_ncols];
      }

      for (int j =1; j <size; j++){
        int remainder = ny%size;
        if(j<remainder || remainder == 0){
          MPI_Recv(recvbuf, local_ncols, MPI_FLOAT, j, tag, MPI_COMM_WORLD, &status);
          for(int jj=0;jj<local_ncols;jj++){
              image[jj + (i-1)*local_ncols + j*(local_nrows-2)*local_ncols] = recvbuf[jj];

          }
        } else {
          if(i <local_nrows-2){
            MPI_Recv(recvbuf, local_ncols, MPI_FLOAT, j, tag, MPI_COMM_WORLD, &status);
            for(int jj = 0; jj<local_ncols;jj++)
              image[jj + (i-1)*local_ncols + (remainder)*(local_nrows-2)*local_ncols + (j-(remainder))*(local_nrows-3)*local_ncols] = recvbuf[jj];
          }
        }
      }

    }
  } else {
    for (int i =1; i <local_nrows-1; i++){
      for(int jj=0;jj<local_ncols;jj++)
        sendbuf[jj] = local_gridcurrent[jj + i*local_ncols];
      MPI_Send(sendbuf, local_ncols, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
    }
  }

  //printf("Process %d from host %s from a total of %d processes, has finished\n", rank, hostname, size);
  // Output
  if(rank==MASTER){
    printf("------------------------------------\n");
    printf(" %d CORES USED\n", size);
    printf(" Running for dimensions %dx%d at %d iterations:\n", nx,ny,niters);
    printf(" runtime: %lf s\n", toc-tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, nx, ny, image);
  }

  MPI_Finalize();

  free(image);

  return EXIT_SUCCESS;
}

void startProcess(const int local_ncols, const int local_nrows, float *sendbuf, float *recvbuf, const int up, const int down, const int rank, const int size, int tag, float *local_gridcurrent, float *local_gridnext, MPI_Status status){
  //Exchange local grid halo rows
  //first send up and receive down
  for(int jj=0;jj<local_ncols;jj++)
    sendbuf[jj] = local_gridcurrent[jj+local_ncols];

  MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, up, tag,
               recvbuf, local_ncols, MPI_FLOAT, down, tag,
               MPI_COMM_WORLD, &status);
  if(rank!=(size-1))
    for(int jj=0;jj<local_ncols;jj++)
      local_gridcurrent[jj+(local_nrows-1)*local_ncols] = recvbuf[jj];

  //now send down receive up
  for(int jj=0;jj<local_ncols;jj++)
    sendbuf[jj] = local_gridcurrent[jj+(local_nrows-2)*local_ncols];

  MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, down, tag,
               recvbuf, local_ncols, MPI_FLOAT, up, tag,
               MPI_COMM_WORLD, &status);
  if(rank!=MASTER)
    for(int jj=0;jj<local_ncols;jj++)
      local_gridcurrent[jj] = recvbuf[jj];

  stencil(local_nrows, local_ncols, local_gridcurrent, local_gridnext);
}

void stencil(const int ny, const int nx, float * restrict image, float * restrict tmp_image) {

  for(int i=1; i<ny-1; i++){
    tmp_image[i*nx] = image[i*nx] * 0.6 +
                    (image[i*nx+1] +
                     image[(i-1)*nx] +
                     image[(i+1)*nx])*0.1;
    for(int j=1; j<nx-1; j++){
        tmp_image[j + i*nx] = image[j + i*nx]*0.6 +
                        (image[j-1 + i*nx] +
                         image[j-nx + i*nx] +
                         image[j+nx + + i*nx] +
                         image[j+1 + i*nx])*0.1;
    }
    tmp_image[(i+1)*nx -1] = image[(i+1)*nx -1]*0.6 +
                    (image[(i+1)*nx -2] +
                     image[i*nx - 1] +
                     image[(i+2)*nx-1])*0.1;
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

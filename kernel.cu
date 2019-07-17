// UID : 180128022

// imports
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "acp18rs"		//my user name

void print_help();


typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE; // type of modes

MODE execution_mode = CPU;

// declaration
unsigned char **red_img;
unsigned char **green_img;
unsigned char **blue_img;
char* str_cat(char *s1, char *s2);


// function call for command line process
int process_command_line(int argc, char *argv[], int *c, char *infile, char *outfile, char *PPM);
// function call for reading images
int in_img(char *infile, int c, int *width, int *height, char *header, char *outfile, char *PPM);
// function call for reading headers
int read_header(FILE *file, int c, int *width, int *height, char *header, char *outfile, char *PPM, char *format);
// function call to launch CPU mode
void CPU_launcher(int c, int width, int height, int *red_ave_host, int *green_ave_host, int *blue_ave_host);
// function call to launch OPENMP mode
void OPENMP_launcher(int c, int width, int height, int *red_ave_host, int *green_ave_host, int *blue_ave_host);
// function call to convert a 2D array to a 1D array
void converter_1D(unsigned char* red_CPU, unsigned char* green_CPU, unsigned char* blue_CPU, int width, int height);
// function call to launch CUDA mode
void CUDA_launcher(int c, int width, int height, unsigned char* red_CPU, unsigned char* green_CPU, unsigned char* blue_CPU, int *red_ave_host, int *green_ave_host, int *blue_ave_host);
// function call to convert 1D array back to 2D array
void converter_2D(unsigned char* red_CPU, unsigned char* green_CPU, unsigned char* blue_CPU, int width, int height);
// function call to check CUDA error if any
void errorCUDA(const char *msg);
// function call for writing images
int out_img(int width, int height, char *PPM, char *header, char *outfile);


// function to read input images
int in_img(char *infile, int c, int *width, int *height, char *header, char *outfile, char *PPM) {
	// creating a variable of type FILE* and opening it
	FILE *file = fopen(infile, "rb");
	// testing to check the existence of the file
	if (file == NULL) {
		fprintf(stderr, "Error: Can not find the input file. \n");
		return FAILURE;
	}

	// dynamically allocating memory using a char array
	char *format = (char*)malloc(3 * sizeof(*format));
	// handling header-read failure
	if (read_header(file, c, width, height, header, outfile, PPM, format) == FAILURE) {
		fprintf(stderr, "Error: Can not read header...\n");
		return FAILURE;
	}

	// initialising two-dimensional dynamic int arrays and allocating memory to the first dimension
	red_img = (unsigned char **)malloc(sizeof(unsigned char *)*(*height));
	green_img = (unsigned char **)malloc(sizeof(unsigned char *)*(*height));
	blue_img = (unsigned char **)malloc(sizeof(unsigned char *)*(*height));
	int k;
	// allocating memory to the second dimension
	for (k = 0; k < *height; k++) {
		*(red_img + k) = (unsigned char *)malloc(sizeof(unsigned char)*(*width));
		*(green_img + k) = (unsigned char *)malloc(sizeof(unsigned char)*(*width));
		*(blue_img + k) = (unsigned char *)malloc(sizeof(unsigned char)*(*width));
	}

	// dynamically allocating memory 
	unsigned char *inputs = (unsigned char *)malloc(sizeof(unsigned char)*(*width)*(*height) * 3);

	// reading input in PPM_BINARY format (P6)
	if (strcmp(format, "P6") == 0) {
		// reading all the binary content
		fread(inputs, sizeof(unsigned char), 3 * (*width) * (*height), file);
	}
	// reading input in PPM_PLAIN_TEXT format (P3)
	else if (strcmp(format, "P3") == 0) {
		unsigned char *buf = (unsigned char *)malloc(sizeof(unsigned char) * 1);
		int ticker = 0;
		// while loop acting as a checker
		while (fscanf(file, "%u", &buf) == 1) {
			inputs[ticker] = (unsigned char)buf;
			ticker++;
		}
	}
	// initialising variables
	int i = 0;
	int h = -1;
	int w = 0;
	while (i < (*width)*(*height) * 3) {
		if (i % (*width * 3) == 0) {
			h++;
			w = 0;
		}
		// doing for red
		if (i % 3 == 0) {
			red_img[h][w] = inputs[i];

		}
		// doing for green
		else if (i % 3 == 1) {
			green_img[h][w] = inputs[i];

		}
		// doing for blue
		else {
			blue_img[h][w] = inputs[i];
			w++;
		}
		i++;
	}
	// closing the file
	fclose(file);
	// freeing the allocated memory
	free(inputs);
	free(format);

	return SUCCESS;
}

// function to write output to files
int out_img(int width, int height, char *PPM, char *header, char *outfile) {
	// conditions to match the criterion


	if (strcmp(PPM, "PPM_BINARY") == 0) {
		// creating a variable of type FILE* and opening it
		FILE *file = fopen(outfile, "wb");
		// testing to check the existence of the file
		if (file == NULL) {
			fprintf(stderr, "Error: Can't find the output file...\n");
			return FAILURE;
		}
		// writing header information
		fprintf(file, "%s", header);
		// dynamically allocating memory
		unsigned char *outputs = (unsigned char *)malloc(sizeof(unsigned char)*width*height * 3);
		// initialising variables
		int i = 0;
		int h = 0;
		int w = 0;
		for (h = 0; h < height; h++) {
			for (w = 0; w < width; w++) {
				// doing for red
				outputs[i] = red_img[h][w];
				i++;
				// doing for green
				outputs[i] = green_img[h][w];
				i++;
				// doing for blue
				outputs[i] = blue_img[h][w];
				i++;
			}
		}
		// writing all the information
		fwrite(outputs, sizeof(unsigned char), 3 * width*height, file);
		// closing the file
		fclose(file);
		// freeing the allocated memory
		free(outputs);

	}
	else if (strcmp(PPM, "PPM_PLAIN_TEXT") == 0) {
		// creating a variable of type FILE* and opening it
		FILE *file = fopen(outfile, "w");
		// testing to check the existence of the file
		if (file == NULL) {
			fprintf(stderr, "Error: Can't find the output file...\n");
			return FAILURE;
		}
		// writing header information
		fputs(header, file);
		int h, w;
		char out_string[4];
		for (h = 0; h < height; h++) {
			for (w = 0; w < width; w++) {
				// doing for red
				sprintf(out_string, "%u", red_img[h][w]);
				// writing string to a stream
				fputs(out_string, file);
				// writing character to a stream
				fputc(' ', file);
				// doing for green
				sprintf(out_string, "%u", green_img[h][w]);
				fputs(out_string, file);
				fputc(' ', file);
				// doing for blue
				sprintf(out_string, "%u", blue_img[h][w]);
				fputs(out_string, file);

				if (w == (width - 1))
					continue;
				fputc('\t', file);
			}
			// shifting to a new line
			if (h == (height - 1))
				continue;
			fputc('\n', file);
		}
		// closing file
		fclose(file);

	}
	return SUCCESS;
}

// function to read header information
int read_header(FILE *file, int c, int *width, int *height, char *header, char *outfile, char *PPM, char *format) {
	//initialising variables
	char input[1024] = "";
	char *size = "";

	// reading header infomation
	while (1) {
		// exit condition for reading to EOF
		if (fgets(input, sizeof(input), file) == NULL) {
			return FAILURE;
		}
		// exit condition for reading to the end line of header
		if (strncmp(input, "255", 3) == 0) {
			size = str_cat(size, input);
			break;
		}
		// condition for either P6 or P3
		if (strncmp(input, "P6", 2) == 0) {
			strcpy(format, "P6");
		}
		else if (strncmp(input, "P3", 2) == 0) {
			strcpy(format, "P3");
		}
		// skipping if reading to the command line
		else if (strncmp(input, "#", 1) == 0) {
			continue;
		}
		// width is the first number and height is the second one
		else {
			size = str_cat(size, input);
			// handling if width is not assigned
			if (*width == 0) {
				*width = atoi(input);
			}
			else {
				*height = atoi(input);
			}
		}
	}
	// checking if the value of c is greater than width or height of the image
	if (c > *width || c > *height) {
		fprintf(stderr, "Error: The value of c has a higher value than either width or height of the original image. \n");
		return FAILURE;
	}


	char* heading = "";
	// Header information of the output file
	if (strcmp(PPM, "PPM_BINARY") == 0) {
		heading = str_cat(heading, "P6\n");
	}
	else if (strcmp(PPM, "PPM_PLAIN_TEXT") == 0) {
		heading = str_cat(heading, "P3\n");
	}
	// information to be written to the output files
	heading = str_cat(heading, "# COM6521 Assignment2 - ");
	heading = str_cat(heading, outfile);
	heading = str_cat(heading, "\n");
	heading = str_cat(heading, size);

	// copying the value of string
	strcpy(header, heading);
	// freeing memory
	free(heading);

	return SUCCESS;
}


char* str_cat(char *s1, char *s2) {
	// allocating memory space and adding 1 for the zero-terminator
	char *result = (char*)malloc(strlen(s1) + strlen(s2) + 1);
	if (result == NULL)
		exit(1);
	// copying string values
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}


// function for launching CPU mode
void CPU_launcher(int c, int width, int height, int *red_ave_host, int *green_ave_host, int *blue_ave_host) {
	clock_t begin, end;
	float seconds;

	// starting timing here
	begin = clock();

	// initialise the results
	*red_ave_host = 0, *green_ave_host = 0, *blue_ave_host = 0;
	int i = 0;
	int j = 0;
	int k = 0;
	int l = 0;
	for (i = 0; i < height; i += c) {
		for (j = 0; j < width; j += c) {
			int red_added = 0;
			int green_added = 0;
			int blue_added = 0;
			int ticker = 0;
			for (k = i; k < (i + c) && k < height; k++) {
				for (l = j; l < (j + c) && l < width; l++) {
					ticker++;
					red_added += red_img[k][l];
					green_added += green_img[k][l];
					blue_added += blue_img[k][l];
				}
			}
			for (k = i; k < (i + c) && k < height; k++) {
				for (l = j; l < (j + c) && l < width; l++) {
					red_img[k][l] = (int)red_added / ticker;
					green_img[k][l] = (int)green_added / ticker;
					blue_img[k][l] = (int)blue_added / ticker;
				}
			}
			*red_ave_host += red_added;
			*green_ave_host += green_added;
			*blue_ave_host += blue_added;
		}
	}
	*red_ave_host = (int)*red_ave_host / (height * width);
	*green_ave_host = (int)*green_ave_host / (height * width);
	*blue_ave_host = (int)*blue_ave_host / (height * width);

	// end timing here
	end = clock();
	seconds = (end - begin) * 1000 / (float)CLOCKS_PER_SEC;
	printf("CPU mode execution time took %d s and %d ms\n", (int)seconds / 1000, (int)seconds % 1000);
}

// function for launching OPENMP mode
void OPENMP_launcher(int c, int width, int height, int *red_ave_host, int *green_ave_host, int *blue_ave_host) {
	clock_t begin, end;
	float seconds;

	// starting timing here
	begin = clock();

	// initialise the results
	int red = 0;
	int green = 0;
	int blue = 0;
	int i;
#pragma omp parallel for
	for (i = 0; i < height; i += c) {
		int j = 0;
		//pragma omp parallel for
#pragma omp parallel for reduction(+: red, green, blue)
		for (j = 0; j < width; j += c) {
			int k, l, red_added = 0, green_added = 0, blue_added = 0, ticker = 0;
			for (k = i; k < (i + c) && k < height; k++) {
				for (l = j; l < (j + c) && l < width; l++) {
					ticker++;
					red_added += red_img[k][l];
					green_added += green_img[k][l];
					blue_added += blue_img[k][l];
				}
			}
			for (k = i; k < (i + c) && k < height; k++) {
				for (l = j; l < (j + c) && l < width; l++) {
					red_img[k][l] = (int)red_added / ticker;
					green_img[k][l] = (int)green_added / ticker;
					blue_img[k][l] = (int)blue_added / ticker;
				}
			}
			//pragma omp critical
			//{
			red += red_added;
			green += green_added;
			blue += blue_added;
			//}
		}
	}
	red = (int)red / (height * width);
	green = (int)green / (height * width);
	blue = (int)blue / (height * width);

	*red_ave_host = red;
	*green_ave_host = green;
	*blue_ave_host = blue;

	// end timing here
	end = clock();
	seconds = (end - begin) * 1000 / (float)CLOCKS_PER_SEC;
	printf("OPENMP mode execution time took %d s and %d ms\n", (int)seconds / 1000, (int)seconds % 1000);
}

// calling from GPU to be run on the GPU
__device__ int red_average_dev, green_average_dev, blue_average_dev;
// calling from CPU to be run on the GPU
__global__ void Kernel(unsigned char* red_GPU, unsigned char* green_GPU, unsigned char* blue_GPU, const int width, const int height, const int c) {

	// using shared memory as an optimisation method
	__shared__ int red_data, green_data, blue_data;
	// declaring variables
	unsigned int off;
	unsigned int i;
	int block_x = -1;
	int block_y = -1;
	int rem_x = 0;
	int rem_y = 0;
	int red_added = 0;
	int green_added = 0;
	int blue_added = 0;

	// conditions for appropriating the dimensions of the input image based on the 'c' value
	if (height % c != 0) {
		block_y = height / c;
		rem_y = height % c;
	}
	if (width % c != 0) {
		block_x = width / c;
		rem_x = width % c;
	}

	if (blockIdx.x != block_x || threadIdx.x < rem_x) {
		for (i = 0; i < blockDim.x; i++) {
			if (blockIdx.y == block_y && i >= rem_y)
				continue;

			off = (blockIdx.y * blockDim.x * width) + (blockIdx.x * blockDim.x + threadIdx.x) + width * i;
			red_added += red_GPU[off];
			green_added += green_GPU[off];
			blue_added += blue_GPU[off];
		}
		// using atomicAdd as an optimisation measure to prevent Race Conditions
		atomicAdd(&red_data, red_added);
		atomicAdd(&green_data, green_added);
		atomicAdd(&blue_data, blue_added);
	}

	// ensuring correct results when parallel threads cooperate
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(&red_average_dev, red_data);
		atomicAdd(&green_average_dev, green_data);
		atomicAdd(&blue_average_dev, blue_data);

		// necessary operation conditions
		if (blockIdx.x == block_x && blockIdx.y == block_y) {
			red_data /= rem_x * rem_y;
			green_data /= rem_x * rem_y;
			blue_data /= rem_x * rem_y;
		}
		else if (blockIdx.x != block_x && blockIdx.y == block_y) {
			red_data /= c * rem_y;
			green_data /= c * rem_y;
			blue_data /= c * rem_y;
		}
		else if (blockIdx.x == block_x && blockIdx.y != block_y) {
			red_data /= rem_x * c;
			green_data /= rem_x * c;
			blue_data /= rem_x * c;
		}
		else {
			red_data /= c * c;
			green_data /= c * c;
			blue_data /= c * c;
		}
	}

	// synchronising the threads
	__syncthreads();

	// assigning the values back based on the below conditions
	if (blockIdx.x != block_x || threadIdx.x < rem_x) {
		for (i = 0; i < blockDim.x; i++) {
			if (blockIdx.y == block_y && i >= rem_y) {
				continue;
			}
			off = (blockIdx.y * blockDim.x * width) + (blockIdx.x * blockDim.x + threadIdx.x) + width * i;
			red_GPU[off] = red_data;
			green_GPU[off] = green_data;
			blue_GPU[off] = blue_data;
		}
	}

}

// converting 2D to 1D before assigning the values and updating the counter
void converter_1D(unsigned char* red_CPU, unsigned char* green_CPU, unsigned char* blue_CPU, int width, int height) {
	int ticker = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			red_CPU[ticker] = red_img[i][j];
			green_CPU[ticker] = green_img[i][j];
			blue_CPU[ticker] = blue_img[i][j];
			ticker++;
		}
	}
}


// function for launching CUDA mode
void CUDA_launcher(int c, int width, int height, unsigned char* red_CPU, unsigned char* green_CPU, unsigned char* blue_CPU, int *red_ave_host, int *green_ave_host, int *blue_ave_host) {
	cudaEvent_t start, stop;
	float seconds;



	// creating timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// starting timing here
	cudaEventRecord(start, 0);

	*red_ave_host = 0, *green_ave_host = 0, *blue_ave_host = 0;

	// declaring GPU pointers
	unsigned char *red_GPU;
	unsigned char *green_GPU;
	unsigned char *blue_GPU;

	int block_x;
	int block_y;

	// allocating CUDA memory
	cudaMalloc((void**)&red_GPU, sizeof(unsigned char)*(width)*(height));
	cudaMalloc((void**)&green_GPU, sizeof(unsigned char)*(width)*(height));
	cudaMalloc((void**)&blue_GPU, sizeof(unsigned char)*(width)*(height));
	// throwing an error upon finding issues with memory allocation
	errorCUDA("CUDA Memory Allocation Error");

	// coping data to a symbol on the device for red, green and blue
	cudaMemcpyToSymbol(red_average_dev, red_ave_host, sizeof(int));
	cudaMemcpyToSymbol(green_average_dev, green_ave_host, sizeof(int));
	cudaMemcpyToSymbol(blue_average_dev, blue_ave_host, sizeof(int));

	// copying data from the host to the device for red, green and blue
	cudaMemcpy(red_GPU, red_CPU, sizeof(unsigned char)*(width)*(height), cudaMemcpyHostToDevice);
	cudaMemcpy(green_GPU, green_CPU, sizeof(unsigned char)*(width)*(height), cudaMemcpyHostToDevice);
	cudaMemcpy(blue_GPU, blue_CPU, sizeof(unsigned char)*(width)*(height), cudaMemcpyHostToDevice);
	// raising an error in case of an issue with the above processes
	errorCUDA("CUDA Error while copying memory from host to device!");

	// assigning values to block_x and block_y based on the divisibility of image dimensions with 'c'
	if (width % c == 0) {
		block_x = width / c;
	}
	else {
		block_x = width / c + 1;
	}

	if (height % c == 0) {
		block_y = height / c;
	}
	else {
		block_y = height / c + 1;
	}

	//defining cuda layout and execution
	dim3 threadsPerBlock(c, 1, 1);
	dim3 blocksPerGrid(block_x, block_y, 1);

	// calling from CPU to be run on the GPU
	Kernel << <blocksPerGrid, threadsPerBlock >> > (red_GPU, green_GPU, blue_GPU, width, height, c);

	// copying data from the given symbol on the device
	cudaMemcpyFromSymbol(red_ave_host, red_average_dev, sizeof(int));
	cudaMemcpyFromSymbol(green_ave_host, green_average_dev, sizeof(int));
	cudaMemcpyFromSymbol(blue_ave_host, blue_average_dev, sizeof(int));

	// copying data back to the host
	cudaMemcpy(red_CPU, red_GPU, sizeof(unsigned char)*(width)*(height), cudaMemcpyDeviceToHost);
	cudaMemcpy(green_CPU, green_GPU, sizeof(unsigned char)*(width)*(height), cudaMemcpyDeviceToHost);
	cudaMemcpy(blue_CPU, blue_GPU, sizeof(unsigned char)*(width)*(height), cudaMemcpyDeviceToHost);
	errorCUDA("CUDA Error while copying memory from device to host!");

	*red_ave_host /= width * height;
	*green_ave_host /= width * height;
	*blue_ave_host /= width * height;

	// end timing here
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	errorCUDA("Timer!");
	printf("CUDA mode execution time took %d s and %d ms\n", (int)seconds / 1000, (int)seconds % 1000);

	// Freeing GPU memory
	cudaFree(red_GPU);
	cudaFree(green_GPU);
	cudaFree(blue_GPU);

	// destroying the event object
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// destroying all the allocations and resetting all states on the device in the current process
	cudaDeviceReset();

}

// function to convert 1D array back to 2D array
void converter_2D(unsigned char* red_CPU, unsigned char* green_CPU, unsigned char* blue_CPU, int width, int height) {
	long long ticker = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			red_img[i][j] = red_CPU[ticker];
			green_img[i][j] = green_CPU[ticker];
			blue_img[i][j] = blue_CPU[ticker];
			ticker++;
		}
	}
}

// function for handling CUDA errors
void errorCUDA(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}



// main function
int main(int argc, char *argv[]) {
	// declaring variables
	int c = 0;
	int width = 0;
	int height = 0;
	int red_ave_host = 0;
	int green_ave_host = 0;
	int blue_ave_host = 0;
	// allocating memory
	char *infile = (char*)malloc(100 * sizeof(*infile));
	char *outfile = (char*)malloc(100 * sizeof(*outfile));
	char *PPM = (char*)malloc(15 * sizeof(*PPM));
	char *header = (char*)malloc(1024 * sizeof(*header));

	// handling a command line error
	if (process_command_line(argc, argv, &c, infile, outfile, PPM) == FAILURE) {
		return 1;
	}
	// printing the necessary information related to the input file
	printf("The size of c is: %d\n", c);
	printf("The name of the input file is: %s\n", infile);
	printf("The name of the output is: %s\n", outfile);
	printf("The format is: %s\n", PPM);

	// reading input image
	if (in_img(infile, c, &width, &height, header, outfile, PPM) == FAILURE) {
		return 1;
	}
	printf("The image width is: %d\n", width);
	printf("The image height is: %d\n", height);

	// allocating memory
	unsigned char *red_CPU = (unsigned char *)malloc(sizeof(*red_CPU)*(width)*(height));
	unsigned char *green_CPU = (unsigned char *)malloc(sizeof(*green_CPU)*(width)*(height));
	unsigned char *blue_CPU = (unsigned char *)malloc(sizeof(*blue_CPU)*(width)*(height));

	// execute the mosaic filter based on the mode
	switch (execution_mode) {
	case (CPU): {
		// Run code using CPU
		printf("Executing in CPU Mode... \n");
		// calculate the average colour value
		CPU_launcher(c, width, height, &red_ave_host, &green_ave_host, &blue_ave_host);
		// Output the average colour value for the image
		printf("CPU Average image colour red = %d, green = %d, blue = %d \n", red_ave_host, green_ave_host, blue_ave_host);

		break;
	}
	case (OPENMP): {
		// Run code using OPENMP
		printf("Executing in OPENMP Mode... \n");
		// calculate the average colour value
		OPENMP_launcher(c, width, height, &red_ave_host, &green_ave_host, &blue_ave_host);
		// Output the average colour value for the image
		printf("OPENMP Average image colour red = %d, green = %d, blue = %d \n", red_ave_host, green_ave_host, blue_ave_host);

		break;
	}
	case (CUDA): {
		// Run code using CUDA
		printf("Executing in CUDA Mode... \n");
		converter_1D(red_CPU, green_CPU, blue_CPU, width, height);

		// calculate the average colour value
		CUDA_launcher(c, width, height, red_CPU, green_CPU, blue_CPU, &red_ave_host, &green_ave_host, &blue_ave_host);

		// Output the average colour value for the image
		printf("CUDA Average image colour red = %d, green = %d, blue = %d \n", red_ave_host, green_ave_host, blue_ave_host);
		converter_2D(red_CPU, green_CPU, blue_CPU, width, height);

		break;
	}
	case (ALL): {
		// Run code using CPU
		printf("Executing in CPU Mode... \n");
		// CPU: calculate the average colour value
		CPU_launcher(c, width, height, &red_ave_host, &green_ave_host, &blue_ave_host);
		// CPU: output the average colour value for the image
		printf("CPU Average image colour red = %d, green = %d, blue = %d \n\n", red_ave_host, green_ave_host, blue_ave_host);

		// Run code using OPENMP
		printf("Executing in OPENMP Mode... \n");
		// OPENMP: calculate the average colour value
		OPENMP_launcher(c, width, height, &red_ave_host, &green_ave_host, &blue_ave_host);
		// OPENMP: output the average colour value for the image
		printf("OPENMP Average image colour red = %d, green = %d, blue = %d \n\n", red_ave_host, green_ave_host, blue_ave_host);

		// Run code using CUDA
		printf("Executing in CUDA Mode... \n");
		converter_1D(red_CPU, green_CPU, blue_CPU, width, height);
		// CUDA: calculate the average colour value
		CUDA_launcher(c, width, height, red_CPU, green_CPU, blue_CPU, &red_ave_host, &green_ave_host, &blue_ave_host);
		// CUDA: output the average colour value for the image
		printf("CUDA Average image colour red = %d, green = %d, blue = %d \n\n", red_ave_host, green_ave_host, blue_ave_host);
		converter_2D(red_CPU, green_CPU, blue_CPU, width, height);

		break;
	}
	}

	// saving the output image file (from last executed mode)
	if (out_img(width, height, PPM, header, outfile) == FAILURE) {
		return 1;
	}

	// free memory
	int k;
	for (k = 0; k < height; k++) {
		free(red_img[k]);
		free(green_img[k]);
		free(blue_img[k]);
	}

	// free CPU memory
	free(red_CPU);
	free(green_CPU);
	free(blue_CPU);
	free(red_img);
	free(green_img);
	free(blue_img);
	free(infile);
	free(outfile);
	free(PPM);
	free(header);

	return 0;
}


void print_help() {
	printf("mosaic_%s C M -i infile -o outfile [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		"\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		"\t               ALL. The mode specifies which version of the simulation\n"
		"\t               code should execute. ALL should execute each mode in\n"
		"\t               turn.\n");
	printf("\t-i infile  Specifies an input image file\n");
	printf("\t-o outfile Specifies an output image file which will be used\n"
		"\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f PPM  PPM image output format either PPM_BINARY (default) or \n"
		"\t               PPM_PLAIN_TEXT\n ");
}



int process_command_line(int argc, char *argv[], int *c, char *infile, char *outfile, char *PPM) {

	*c = atoi(argv[1]);


	//TODO: read in the mode
	if (strcmp("CPU", argv[2]) == 0)
		execution_mode = CPU;
	else if (strcmp("OPENMP", argv[2]) == 0)
		execution_mode = OPENMP;
	// taking in CUDA mode
	else if (strcmp("CUDA", argv[2]) == 0)
		execution_mode = CUDA;
	else if (strcmp("ALL", argv[2]) == 0)
		execution_mode = ALL;


	//TODO: read in the input image name
	if (strcmp("-i", argv[3]) == 0) {
		strcpy(infile, argv[4]);
	}
	else {
		fprintf(stderr, "Error: Illegal arguments. \n");
		print_help();
		return FAILURE;
	}

	//TODO: read in the output image name
	if (strcmp("-o", argv[5]) == 0) {
		strcpy(outfile, argv[6]);
	}
	else {
		fprintf(stderr, "Error: Illegal arguments. \n");
		print_help();
		return FAILURE;
	}

	//TODO: read in any optional part 3 arguments
	if (argc == 9) {
		if (strcmp("-f", argv[7]) == 0) {
			strcpy(PPM, argv[8]);
		}
		else {
			fprintf(stderr, "Error: Illegal arguments. \n");
			print_help();
			return FAILURE;
		}
	}
	else {
		// assigning PPM_BINARY as the default format
		strcpy(PPM, "PPM_BINARY");
	}

	return SUCCESS;
}


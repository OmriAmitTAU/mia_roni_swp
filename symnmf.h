#ifndef SYMNMF_H
#define SYMNMF_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define EPSILON 0.0001

/* Function to initialize a zeros matrix */
double **initialize_matrix(int numRows, int numCols);

/* Function to transpose a given matrix */
double **transpose(double **mat, int numRows, int numCols);

/* Function for matrix multiplication */
double **matrix_multiplication(double **mat1, int rows1, int cols1, double **mat2, int rows2, int cols2);

/* Function to calculate Frobenius distance between the given matrices */
double frobidean_distance(double **mat1, double **mat2, int rows, int cols);

/* Function to calculate Euclidean distance between two given vectors */
double euclidean_distance(double *vec1, double *vec2, int dim);

/* Helper function to free allocated memory for a matrix */
void free_matrix(double **matrix, int n);

/* Function to calculate sym function */
double **symc(double **points, int n, int d);

/* Function for ddg */
double **ddgc(double **points, int n, int d);

/* Function to calculate norm */
double **normc(double **points, int n, int d);

/* Function for iteration of symnmf */
double **calc(double **H, double **W, int n, int k);

/* Function to perform the symnmf */
double **symnmfc(double **H, double **W, int n, int k);

/* Function that for each point returns its cluster index */
int *analysisc(double **H, int n, int k);

/* Helper function to read file and count rows and columns */
void read_file_dimensions(char *file_name, int *n, int *d);

/* Helper function to read data from file into matrix */
double **read_data(char *file_name, int n, int d);

/* Helper function to initialize the matrix based on the goal */
double **initialize_matrix_goal(double **data, int n, int d, char *goal);

/* Helper function to print the matrix */
void print_matrix(double **matrix, int n);

#endif /* SYMNMF_H */

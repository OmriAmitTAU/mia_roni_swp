#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"

/* function to initialize zeros matrix */
double **initialize_matrix(int numRows, int numCols)
{
    double **mat = (double **)malloc(numRows * sizeof(double *));
    int rowIndex = 0;
    int colIndex = 0;

    for (rowIndex = 0; rowIndex < numRows; rowIndex++)
    {
        mat[rowIndex] = (double *)malloc(numCols * sizeof(double));
        for (colIndex = 0; colIndex < numCols; colIndex++)
        {
            mat[rowIndex][colIndex] = 0.0;
        }
    }
    return mat;
}

/* function to transpose given matrix */
double **transpose(double **mat, int numRows, int numCols)
{
    double **transposed = (double **)malloc(numCols * sizeof(double *));
    int rowIndex = 0;
    int colIndex = 0;

    for (colIndex = 0; colIndex < numCols; colIndex++)
    {
        transposed[colIndex] = (double *)malloc(numRows * sizeof(double));
        for (rowIndex = 0; rowIndex < numRows; rowIndex++)
        {
            transposed[colIndex][rowIndex] = mat[rowIndex][colIndex];
        }
    }
    return transposed;
}

/* Function for matrix multiplication */
double **matrix_multiplication(double **mat1, int rows1, int cols1, double **mat2, int rows2, int cols2)
{
    double **output;
    int r1_index;
    int c2_index;
    int c1_index;

    if (cols1 != rows2)
    {
        return NULL;
    }

    output = initialize_matrix(rows1, cols2);

    for (r1_index = 0; r1_index < rows1; r1_index++)
    {
        for (c2_index = 0; c2_index < cols2; c2_index++)
        {
            for (c1_index = 0; c1_index < cols1; c1_index++)
            {
                output[r1_index][c2_index] += mat1[r1_index][c1_index] * mat2[c1_index][c2_index];
            }
        }
    }
    return output;
}

/* function to calculate Frobenius distance between the given matrices */
double frobidean_distance(double **mat1, double **mat2, int rows, int cols)
{
    int r;
    int c;
    double distance = 0.0;
    for (r = 0; r < rows; r++)
    {
        for (c = 0; c < cols; c++)
        {
            distance += pow(mat1[r][c] - mat2[r][c], 2);
        }
    }
    distance = sqrt(distance);
    return distance;
}

/* function to calculate Euclidean distance between two given vectors */
double euclidean_distance(double *vec1, double *vec2, int dim)
{
    int i;
    double distance = 0.0;
    for (i = 0; i < dim; i++)
    {
        distance += pow(vec1[i] - vec2[i], 2);
    }
    return distance;
}

/* Helper function to free allocated memory for a matrix */
void free_matrix(double **matrix, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

/* function to calculate sym function */
double **symc(double **points, int n, int d)
{
    int i;
    int j;
    double **sym_matrix = initialize_matrix(n, n);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < i; j++)
        {
            sym_matrix[i][j] = exp(-0.5 * euclidean_distance(points[i], points[j], d));
            sym_matrix[j][i] = sym_matrix[i][j];
        }
        sym_matrix[i][i] = 0;
    }
    return sym_matrix;
}

/* function for ddg */
double **ddgc(double **points, int n, int d)
{
    int i;
    int j;
    double **C, **output;
    C = symc(points, n, d);
    output = initialize_matrix(n, n);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            output[i][i] += C[i][j];
        }
    }

    /* Free allocated memory */
    free_matrix(C, n);

    return output;
}

/* function to calculate norm */
double **normc(double **points, int n, int d)
{
    double **D = ddgc(points, n, d);
    double **A = symc(points, n, d);
    double **norm_matrix;
    double **temp_matrix;
    int i;

    for (i = 0; i < n; i++)
    {
        D[i][i] = 1.0 / sqrt(D[i][i]);
    }
    /* calculate norm matrix */
    temp_matrix = matrix_multiplication(D, n, n, A, n, n);
    norm_matrix = matrix_multiplication(temp_matrix, n, n, D, n, n);

    /* free allocated memory */
    for (i = 0; i < n; i++)
    {
        free(D[i]);
        free(A[i]);
        free(temp_matrix[i]);
    }
    free(D);
    free(A);
    free(temp_matrix);

    return norm_matrix;
}

/* function for iteration of symnmf */
double **calc(double **H, double **W, int n, int k)
{
    int i;
    int j;
    double **WH = matrix_multiplication(W, n, n, H, n, k);
    double **Ht = transpose(H, n, k);
    double **HHt = matrix_multiplication(H, n, k, Ht, k, n);
    double **HHtH = matrix_multiplication(HHt, n, n, H, n, k);
    double **next_H = initialize_matrix(n, k);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < k; j++)
        {
            next_H[i][j] = H[i][j] * (0.5 + 0.5 * (WH[i][j] / HHtH[i][j]));
        }
    }

    /* free allocated memory */
    free_matrix(WH, n);
    free_matrix(Ht, k);
    free_matrix(HHt, n);
    free_matrix(HHtH, n);

    return next_H;
}

/* A function to do the symnmf */
double **symnmfc(double **H, double **W, int n, int k)
{
    int iter, i, j;
    double **next_H = initialize_matrix(n, k);
    iter = 0;
    for (iter = 0; iter < 300; iter ++)
    {
        next_H = calc(H, W, n, k);

        if (pow(frobidean_distance(H, next_H, n, k), 2) < EPSILON)
        {
            return next_H;
        }

        for (i = 0; i < n; i++)
        {
            for (j = 0; j < k; j++)
            {
                H[i][j] = next_H[i][j];
            }
        }
    }

    return next_H;
}

/* function that for each point return its cluster index */
int *analysisc(double **H, int n, int k)
{
    int i;
    int j;
    int *labels = (int*) malloc(n * sizeof(int));

    for (i = 0; i < n; i++)
    {
        /* initialization of max to first value of point */
        double max_val = H[i][0];
        int max_index = 0;

        for (j = 1; j < k; j++)
        {
            /* if new max value then update */
            if (H[i][j] > max_val)
            {
                max_val = H[i][j];
                max_index = j;
            }
        }

        labels[i] = max_index;
    }

    return labels;
}

/* Helper function to read file and count rows and columns */
void read_file_dimensions(char *file_name, int *n, int *d)
{
    FILE *file;
    int ch;

    file = fopen(file_name, "r");
    if (!file)
    {
        exit(1);
    }

    *n = 0;
    *d = 0;
    while ((ch = fgetc(file)) != EOF)
    {
        if (ch == '\n')
        {
            (*n)++;
        }
        else if (ch == ',' && *n == 0)
        {
            (*d)++;
        }
    }
    (*d)++;
    fclose(file);
}

/* Helper function to read data from file into matrix */
double **read_data(char *file_name, int n, int d)
{
    FILE *file;
    double **data;
    int i;
    int j;
    
    file = fopen(file_name, "r");
    if (!file)
    {
        exit(1);
    }

    data = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++)
    {
        data[i] = (double *)malloc(d * sizeof(double));
        for (j = 0; j < d; j++)
        {
            if (fscanf(file, "%lf,", &data[i][j]) != 1)
            {
                exit(1);
            }
        }
    }
    fclose(file);
    return data;
}

/* Helper function to initialize the matrix based on the goal */
double **initialize_matrix_goal(double **data, int n, int d, char *goal)
{
    if (strcmp(goal, "sym") == 0)
    {
        return symc(data, n, d);
    }
    else if (strcmp(goal, "ddg") == 0)
    {
        return ddgc(data, n, d);
    }
    else
    {
        return normc(data, n, d);
    }
}

/* Helper function to print the matrix */
void print_matrix(double **matrix, int n)
{
    int i;
    int j;
    for (i = 0; i < n; i++)
    {
        for ( j = 0; j < n; j++)
        {
            printf("%.4f", matrix[i][j]);
            if (j < n - 1)
            {
                printf(",");
            }
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    char *goal, *file_name;
    double **data, **A;
    int n, d;

    if (argc != 3)
    {
        return 1;
    }

    goal = argv[1];
    file_name = argv[2];

    read_file_dimensions(file_name, &n, &d);
    data = read_data(file_name, n, d);
    A = initialize_matrix_goal(data, n, d, goal);

    print_matrix(A, n);

    free_matrix(data, n);
    free_matrix(A, n);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"


/* Function to initialize a matrix with zeros */
double** initialize_matrix(int rows, int cols) {
    double **matrix = (double **)malloc(rows * sizeof(double *));
    int i, j;
    for (i = 0; i < rows; i++) {
        matrix[i] = (double *)malloc(cols * sizeof(double));
        for (j = 0; j < cols; j++) {
            matrix[i][j] = 0.0;
        }
    }
    return matrix;
}


/* Function to transpose a matrix */
double** transpose(double** matrix, int rows, int cols) {
    double** result = (double **)malloc(cols * sizeof(double *));
    for (int i = 0; i < cols; i++) {
        result[i] = (double *)malloc(rows * sizeof(double));
        for (int j = 0; j < rows; j++) {
            result[i][j] = matrix[j][i];
        }
    }
    return result;
}


/* Function to perform matrix multiplication, gets dimenstions of both matrices */
double** matrix_multiplication(double** mat1,int rows1,int cols1,double** mat2,int rows2,int cols2){
    int i,j,k;
    double** mat_mult;
    mat_mult=initialize_matrix(rows1,cols2);
    if(rows2!=cols1){
        return NULL;
    }
    for(i=0;i<rows1;i++){
        for(j=0;j<cols2;j++){
            for(k=0;k<rows2;k++){
                mat_mult[i][j]+=mat1[i][k]*mat2[k][j];
            }
        }
    }
    return mat_mult;
    }


/* Function to calculate Frobedean distance between two matrices */
double frobidean_distance(double** mat1 , double** mat2, int rows,int cols){
    double d =0.0;
    int i,j;
    for(i=0;i<rows;i++){
    for(j=0; j<cols; j++){
        d += (mat1[i][j]-mat2[i][j])*(mat1[i][j]-mat2[i][j]);
    }
    }
    return sqrt(d);
}


/* Function to calculate Euclidian distance between two vectors(points) */
double euclidean_distance(double *vec1 , double *vec2, int dim){
    double d =0.0;
    int j;
    for(j=0; j<dim; j++){
        d += (vec1[j]-vec2[j])*(vec1[j]-vec2[j]);
    }
    return d;
}


/* Function to free memory allocated for a matrix */
void free_mat(double** matrix){
    free(matrix[0]);
    free(matrix);
}


int main(int argc, char *argv[]) {
    char *goal;
    char *file_name;

    int n = 0;
    int d = 0;
    int ch;
    int i, j; 

    /* check if we have the right amount of args */
    if (argc != 3) {
        printf("An Error Has Occurred");
        return 1;
    }

    goal = argv[1];
    file_name = argv[2];

    double **data;
    double **A;

    /* read the file */
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        printf("An Error Has Occurred");        
        return 1;
    }

    /* count number of points and the dimension of each point  */
    while ((ch = fgetc(file)) != EOF) {
        if (ch == '\n') {
            n++;
        } else if ((ch == ',') && (n == 0)) {
            d++;
        }
    }
    d++;

    rewind(file);

    /* write the file data into a matrix */
    data = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++) {
        data[i] = (double *)malloc(d * sizeof(double));
        for (j = 0; j < d; j++) {
            if (fscanf(file, "%lf,", &data[i][j]) != 1) {
                return 1;
            }
        }
    }

    fclose(file);

    /* activate needed function and print the result matrix */
    if (strcmp(goal, "sym") == 0) {
        A = symc(data, n, d);
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                printf("%.4lf", A[i][j]);
                if (j < n - 1) {
                    printf(",");
                }
            }
            printf("\n");
        }
    } else if (strcmp(goal, "ddg") == 0) {
        A = ddgc(data, n, d);
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                printf("%.4lf", A[i][j]);
                if (j < n - 1) {
                    printf(",");
                }
            }
            printf("\n");
        }
    } else {
        A = normc(data, n, d);
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                printf("%.4lf", A[i][j]);
                if (j < n - 1) {
                    printf(",");
                }
            }
            printf("\n");
        }
    }

    /* Free allocated memory for data and A */
    for (i = 0; i < n; i++) {
        free(data[i]);
        free(A[i]);
    }
    free(data);
    free(A);

    return 0;
}


/* Implementation of sym function */
double** symc(double** points,int n,int d){
    double** sim_mat;
    int i;
    int j;
    sim_mat=initialize_matrix(n,n);
    for(i=0;i<n;i++){
        for(j=0;j<i;j++){
            sim_mat[i][j]=exp(-0.5*euclidean_distance(points[i],points[j],d));
            sim_mat[j][i]= sim_mat[i][j];
        }
        sim_mat[i][i]=0;
    }
    return sim_mat;

}


/* Implementation of ddg function */
double** ddgc(double **points, int n, int d) {
    double **C = symc(points, n, d);
    double **D = initialize_matrix(n,n);
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            D[i][i] += C[i][j];
        }
    }


    /* Free allocated memory for matrix */
    for (i = 0; i < n; i++) {
        free(C[i]);
    }
    free(C);

    return D;
}


/* Implementation of norm function */
double** normc(double **points, int n, int d) {
    double **D = ddgc(points, n, d);
    double **A = symc(points, n, d);
    double** norm;
    double** temp;
    int i;
    for (i = 0; i < n; i++) {
        D[i][i] = 1.0 / sqrt(D[i][i]);
    }

    /* calculation of the norm matrix */
    temp = matrix_multiplication(D,n,n,A,n,n);
    norm = matrix_multiplication(temp,n,n,D,n,n);

    /* Free allocated memory for matrices */
    for (i = 0; i < n; i++) {
        free(D[i]);
        free(A[i]); 
        free(temp[i]);
    }
    free(D);
    free(A); 
    free(temp);

    return norm;
}


/* function to perform an iteration of symnmf */
double** calc(double** H,double** W,int n,int k){
    double** WH;
    double** Ht;
    double** HHt;
    double** HHtH;
    double** next_h;
    int i,j;
    double b=0.5;
    WH=matrix_multiplication(W,n,n,H,n,k);
    Ht = transpose(H,n,k);
    HHt = matrix_multiplication(H,n,k,Ht,k,n);
    HHtH = matrix_multiplication(HHt,n,n,H,n,k);
    next_h=initialize_matrix(n,k);
    for(i=0;i<n;i++){
        for(j=0;j<k;j++){
            next_h[i][j]=H[i][j]*(b+b*(WH[i][j]/HHtH[i][j]));
        }
    }

    /* Free allocated memory for matrices */
    free_mat(WH);
    free_mat(Ht);
    free_mat(HHt);
    free_mat(HHtH);
    return next_h;
}


/* A function to perform the symnmf, returns the required matrix */
double** symnmfc(double** H,double** W,int n,int k){
    int iter=0;
    double eps=0.0001;
    double** next_H;
    int i,j;
    next_H=initialize_matrix(n,k);
    while(iter<300){
        next_H=calc(H,W,n,k);

        /* Check for convergence */
        if(pow(frobidean_distance(H,next_H,n,k),2)<eps){
            return next_H;
        }
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                H[i][j] = next_H[i][j];
            }
        }
        iter++;
    }
    return next_H;

}


/* Implementation of analysis function, return for each point its cluster index*/
int* analysisc(double **H, int n, int k){
    int *labels = malloc(n * sizeof(int));
    int i, j;

    for (i = 0; i < n; i++) {
        /* initialize max as first value of the point */
        double max = H[i][0];  
        int maxindex = 0;
        
        /* check if there is a new max and update */
        for (j = 1; j < k; j++) {
            if (H[i][j] > max) {
                max = H[i][j];
                maxindex = j;
            }
        }

        labels[i] = maxindex;
    }

    return labels;
}


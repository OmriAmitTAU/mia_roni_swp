#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double euclidean_distance(double *a, double *b, int n);
void compute_centroids(double **data, int *labels, double **centroids, int n, int k, int d);
int *assign_labels(double **data, double **centroids, int n, int k, int d);
PyObject *kmeans(double **data, double **centroids, int n, int k, int d, int max_iter, double epsilon);

#define INFINITY (__builtin_inff ())

double euclidean_distance(double *a, double *b, int n) {
    double dist = 0;
    int i;
    for (i = 0; i < n; i++) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(dist);
}

void compute_centroids(double **data, int *labels, double **centroids, int n, int k, int d) { /* d is length of a word */
    int *counts = (int *)malloc(k * sizeof(double));
    int i, j;
    for (i = 0; i < k; i++) {
        counts[i] = 0;
        for (j = 0; j < d; j++) {
            centroids[i][j] = 0;
        }
    }
    for (i = 0; i < n; i++) {
        int label = labels[i];
        counts[label]++;
        for (j = 0; j < d; j++) {
            centroids[label][j] += data[i][j];
        }
    }
    for (i = 0; i < k; i++) {
        if (counts[i] > 0) {
            for (j = 0; j < d; j++) {
                centroids[i][j] /= counts[i];
            }
        }
    }

    free(counts);
}

int *assign_labels(double **data, double **centroids, int n, int k, int d) {
    int *labels, i, j, label;
    labels = malloc(n * sizeof(int));
    for (i = 0; i < n; i++) {
        double min_dist = INFINITY;
        label = -1;
        for (j = 0; j < k; j++) {
            double dist = euclidean_distance(data[i], centroids[j], d);
            if (dist < min_dist) {
                min_dist = dist;
                label = j;
            }
        }
        labels[i] = label;
    }
    return labels;
}


PyObject *kmeans( int k, int n, int d, double **data, int max_iter, double **centroids, double epsilon) {
    double **prev_centroids;
    int i,j, iter, converged;
    int* labels;
    prev_centroids = malloc(k * sizeof(double *));
    for (i = 0; i < k; i++) {
        prev_centroids[i] = malloc(d * sizeof(double));
        for (j = 0; j < d; j++) {
            prev_centroids[i][j] = centroids[i][j];
        }
    }
    labels = NULL;
    for (iter = 0; iter < max_iter; iter++) {
        labels = assign_labels(data, centroids, n, k, d);
        compute_centroids(data, labels, centroids, n, k, d);
        converged = 1;
        for (i = 0; i < k; i++) {
            double dist = euclidean_distance(centroids[i], prev_centroids[i], d);
            if (dist > epsilon) {
                converged = 0;
                break;
            }
        }
        if (converged) {
            break;
        }
        free(labels);
        for (i = 0; i < k; i++) {
            for (j = 0; j < d; j++) {
                prev_centroids[i][j] = centroids[i][j];
            }
        }
    }
    PyObject *centroids_list = PyList_New(k);
    for (i = 0; i < k; i++) {
        PyObject *centroid = PyList_New(d);
        for (j = 0; j < d; j++) {
            PyObject *coordinate = PyFloat_FromDouble(centroids[i][j]);
            PyList_SetItem(centroid, j, coordinate);
        }
        PyList_SetItem(centroids_list, i, centroid);
    }
    for (i = 0; i < k; i++) {
        free(centroids[i]);
        free(prev_centroids[i]);
    }
    free(centroids);
    free(prev_centroids);
    free(labels);
    for (i = 0; i < n; i++) {
        free(data[i]);
    }
    free(data);

    return centroids_list;
}

static PyObject *fit(PyObject *self, PyObject *args) {
    PyObject *data_points_obj;
    PyObject *centroids_obj;
    int K;
    int max_iter;
    double epsilon;

    if (!PyArg_ParseTuple(args, "OOiid", &data_points_obj, &centroids_obj, &K, &max_iter, &epsilon)) {
        return NULL;
    }

    int n = PyList_Size(data_points_obj);
    int d = PyList_Size(PyList_GetItem(data_points_obj, 0));
    double **data_points = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        data_points[i] = (double *)malloc(d * sizeof(double));

        PyObject *data_point_obj = PyList_GetItem(data_points_obj, i);
        for (int j = 0; j < d; j++) {
            PyObject *coordinate_obj = PyList_GetItem(data_point_obj, j);
            data_points[i][j] = PyFloat_AsDouble(coordinate_obj);
        }
    }

    double **centroids = (double **)malloc(K * sizeof(double *));
    for (int i = 0; i < K; i++) {
        centroids[i] = (double *)malloc(d * sizeof(double));

        PyObject *centroid_obj = PyList_GetItem(centroids_obj, i);
        for (int j = 0; j < d; j++) {
            PyObject *coordinate_obj = PyList_GetItem(centroid_obj, j);
            centroids[i][j] = PyFloat_AsDouble(coordinate_obj);
        }
    }
    PyObject *result = kmeans(data_points, centroids, n, K, d, max_iter, epsilon);
    if (result == NULL) {
        printf("An Error Has Occurred");
        return NULL;
    }
    return Py_BuildValue("O", result);
}

static PyMethodDef kmeans_methods[] = {
    {"fit", fit, METH_VARARGS, PyDoc_STR("expects a list of the points, initialized k clusters, K - number of clusters, max iteration number and epsilon.")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef kmeans_module = {PyModuleDef_HEAD_INIT,"kmeanssp","K-means clustering module",-1,kmeans_methods};

PyMODINIT_FUNC PyInit_kmeanssp(void) {
    PyObject *mod;
    mod = PyModule_Create(&kmeans_module);
    if (!mod){
        printf("An Error Has Occurred");
        return NULL;
        }
    return mod;
}

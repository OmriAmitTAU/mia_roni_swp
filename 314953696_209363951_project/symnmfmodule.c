#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include "symnmf.h"


/* Implementation for symn function, expects a matrix and its dimension: matrix, rows, cols */
static PyObject* symnmf_sym(PyObject* self, PyObject* args) {
    (void)self;    
    PyObject* py_data;  
    int rows, cols;   

    /* Parse the arguments from Python */
    if (!PyArg_ParseTuple(args, "Oii", &py_data, &rows, &cols)) {
        return NULL;
    }

    /* Convert the Python input data to a C array of doubles */
    double** data = malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        PyObject* row = PyList_GetItem(py_data, i);
        if (!PyList_Check(row) || PyList_Size(row) != cols) {
            PyErr_SetString(PyExc_ValueError, "Invalid input data");
            return NULL;
        }

        data[i] = malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            data[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    /* Call the 'symc' function from the original symnmf.c file */
    double** result = symc(data, rows, cols);

    /* Create a Python list of lists to represent the result */
    PyObject* py_result = PyList_New(rows);
    for (int i = 0; i < rows; i++) {
        PyObject* py_row = PyList_New(rows);
        for (int j = 0; j < rows; j++) {
            PyList_SET_ITEM(py_row, j, PyFloat_FromDouble(result[i][j]));
        }
        PyList_SET_ITEM(py_result, i, py_row);
    }

    /* Free the allocated memory for 'data' and 'result' */
    for (int i = 0; i < rows; i++) {
        free(data[i]);
        free(result[i]);
    }
    free(data);
    free(result);

    return py_result;
}

/* Implementation for ddg function, expects a matrix and its dimension: matrix, rows, cols */
static PyObject* symnmf_ddg(PyObject* self, PyObject* args) {
    (void)self;   
    PyObject* py_data; 
    int rows, cols;    

    /* Parse the arguments from Python */
    if (!PyArg_ParseTuple(args, "Oii", &py_data, &rows, &cols)) {
        return NULL;
    }

    /* Convert the Python input data to a C array of doubles */
    double** data = malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        PyObject* row = PyList_GetItem(py_data, i);
        if (!PyList_Check(row) || PyList_Size(row) != cols) {
            PyErr_SetString(PyExc_ValueError, "Invalid input data");
            return NULL;
        }

        data[i] = malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            data[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }


    /* Call the 'ddgc' function from the original symnmf.c file */
    double** result = ddgc(data, rows, cols);

    /* Create a Python list of lists to represent the result */
    PyObject* py_result = PyList_New(rows);
    for (int i = 0; i < rows; i++) {
        PyObject* py_row = PyList_New(rows);
        for (int j = 0; j < rows; j++) {
            PyList_SET_ITEM(py_row, j, PyFloat_FromDouble(result[i][j]));
        }
        PyList_SET_ITEM(py_result, i, py_row);
    }

    /* Free the allocated memory for 'data' and 'result' */
    for (int i = 0; i < rows; i++) {
        free(data[i]);
        free(result[i]);
    }
    free(data);
    free(result);

    return py_result;
}

/* Implementation for norm function, expects a matrix and its dimension: matrix, rows, cols */
static PyObject* symnmf_norm(PyObject* self, PyObject* args) {
    (void)self;   
    PyObject* py_data;  
    int rows, cols;  

    /* Parse the arguments from Python */
    if (!PyArg_ParseTuple(args, "Oii", &py_data, &rows, &cols)) {
        return NULL;
    }

    /* Convert the Python input data to a C array of doubles */
    double** data = malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        PyObject* row = PyList_GetItem(py_data, i);
        if (!PyList_Check(row) || PyList_Size(row) != cols) {
            PyErr_SetString(PyExc_ValueError, "Invalid input data");
            return NULL;
        }

        data[i] = malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            data[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    /* Call the 'normc' function from the original symnmf.c file */
    double** result = normc(data, rows, cols);

    /* Create a Python list of lists to represent the result */
    PyObject* py_result = PyList_New(rows);
    for (int i = 0; i < rows; i++) {
        PyObject* py_row = PyList_New(rows);
        for (int j = 0; j < rows; j++) {
            PyList_SET_ITEM(py_row, j, PyFloat_FromDouble(result[i][j]));
        }
        PyList_SET_ITEM(py_result, i, py_row);
    }

    /* Free the allocated memory for 'data' and 'result' */
    for (int i = 0; i < rows; i++) {
        free(data[i]);
        free(result[i]);
    }
    free(data);
    free(result);

    return py_result;
}

/* Implementation for symnmf function, expects: initialized H, norm matrix, dimention (n), number of clusters (k)  */
static PyObject* symnmf_symnmf(PyObject* self, PyObject* args) {
    (void)self;   
    PyObject* py_H;   
    PyObject* py_W;  
    int n, k;   

    /* Parse the arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!ii", &PyList_Type, &py_H, &PyList_Type, &py_W, &n, &k)) {
        return NULL;
    }

    /* Convert the Python input H matrix to a C array of doubles */
    double** H = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        PyObject* row = PyList_GetItem(py_H, i);
        if (!PyList_Check(row) || PyList_Size(row) != k) {
            PyErr_SetString(PyExc_ValueError, "Invalid input H matrix");
            return NULL;
        }

        H[i] = malloc(k * sizeof(double));
        for (int j = 0; j < k; j++) {
            H[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    /* Convert the Python input W matrix to a C array of doubles */
    double** W = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        PyObject* row = PyList_GetItem(py_W, i);
        if (!PyList_Check(row) || PyList_Size(row) != n) {
            PyErr_SetString(PyExc_ValueError, "Invalid input W matrix");
            return NULL;
        }

        W[i] = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            W[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    /* Call the 'symnmfc' function from the original symnmf.c file */
    double** result = symnmfc(H, W, n, k);

    /* Create a Python list of lists to represent the result */
    PyObject* py_result = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyObject* py_row = PyList_New(k);  // Use 'k' here instead of 'n'
        for (int j = 0; j < k; j++) {  // Iterate over 'k' elements
            PyList_SET_ITEM(py_row, j, PyFloat_FromDouble(result[i][j]));
        }
        PyList_SET_ITEM(py_result, i, py_row);
    }


    /* Free the allocated memory for 'H', 'W', and 'result' */
    for (int i = 0; i < n; i++) {
        free(H[i]);
        free(W[i]);
        free(result[i]);
    }
    free(H);
    free(W);
    free(result);

    return py_result;
}


/* Implementation for analysis function, expects: matrix returned from symnmf (final H), dimention (n), number of clusters (k) */
static PyObject* symnmf_analysis(PyObject* self, PyObject* args) {
    (void)self;   
    PyObject* py_H;  
    int n, k;    

    /* Parse the arguments from Python */
    if (!PyArg_ParseTuple(args, "O!ii", &PyList_Type, &py_H, &n, &k)) {
        return NULL;
    }

    /* Convert the Python input H matrix to a C array of doubles */
    double** H = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        PyObject* row = PyList_GetItem(py_H, i);
        if (!PyList_Check(row) || PyList_Size(row) != k) {
            PyErr_SetString(PyExc_ValueError, "Invalid input H matrix");
            return NULL;
        }

        H[i] = malloc(k * sizeof(double));
        for (int j = 0; j < k; j++) {
            H[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    /* Call the 'analysis' function from the original C file */
    int* result = analysisc(H, n, k);

    /* Create a Python list of integers to represent the result */
    PyObject* py_result = PyList_New(n);
    for (int i = 0; i < n; i++) {
        PyList_SET_ITEM(py_result, i, PyLong_FromLong(result[i]));
    }

    /* Free the allocated memory for 'H' and 'result' */
    for (int i = 0; i < n; i++) {
        free(H[i]);
    }
    free(H);
    free(result);

    return py_result;
}

/* List of Python methods in the module - will be called by the names written here */
static PyMethodDef symnmf_methods[] = {
    {"sym", symnmf_sym, METH_VARARGS, "Compute the similarity matrix"},
    {"ddg", symnmf_ddg, METH_VARARGS,"Compute the diagonal degree matrix"},
    {"norm", symnmf_norm, METH_VARARGS, "Compute the normalized similarity matrix"},
    {"symnmf", symnmf_symnmf, METH_VARARGS, "Perform 'symnmf'"},
    {"analysis", symnmf_analysis, METH_VARARGS, "Perform 'analysis'"},
    {NULL, NULL, 0, NULL}
};

/* Module definition, determining its name - mysymnmf */
static struct PyModuleDef symnmf_module = {
    PyModuleDef_HEAD_INIT,
    "mysymnmf",
    NULL,
    -1,
    symnmf_methods,
};


/* Module initialization function */
PyMODINIT_FUNC PyInit_mysymnmf(void) {
    PyObject *module;

    module = PyModule_Create(&symnmf_module);
    if (module == NULL) {
        return NULL;
    }

    return module;
}



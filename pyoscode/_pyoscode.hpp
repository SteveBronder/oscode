#pragma once
#include <array>
#include "_python.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <string>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>

/* Docstrings */
static char module_docstring[] = 
"pyoscode: this module provides an interface for oscode, for solving\
oscillatory ordinary differential equations with the RKWKB method.";

static char solve_docstring[] = "Runs the solver with w, g provided as arrays",
            solve_fn_docstring[] = "Runs the solver with w, g provided as\
            functions";

/* Available functions */
/* Solve with w, g provided as arrays */
static PyObject *_pyoscode_solve(PyObject *self, PyObject *args, PyObject *kwargs);
/* Solve with w, g provided as functions */
static PyObject *_pyoscode_solve_fn(PyObject *self, PyObject *args, PyObject *kwargs);

/* Module interface */
static PyMethodDef module_methods[] = {
    {"solve", (PyCFunction) _pyoscode_solve, METH_VARARGS | METH_KEYWORDS, solve_docstring},
    {"solve_fn", (PyCFunction) _pyoscode_solve_fn, METH_VARARGS | METH_KEYWORDS, solve_fn_docstring},
    {NULL, NULL, 0, NULL}
};
 
#ifdef PYTHON3
static struct PyModuleDef _pyoscodemodule = {
    PyModuleDef_HEAD_INIT,
    "_pyoscode", 
    module_docstring,
    -1,
    module_methods
};
#endif

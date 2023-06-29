#define PY_SSIZE_T_CLEAN
#include "_pyoscode.hpp"
#include "_python.hpp"
#include <oscode/solver.hpp>
#include <oscode/system.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <complex>
#include <numpy/arrayobject.h>
#include <vector>

/* Initialise the module */
#ifdef PYTHON3
PyMODINIT_FUNC PyInit__pyoscode(void) {
  import_array();
  return PyModule_Create(&_pyoscodemodule);
}
#else
PyMODINIT_FUNC init_pyoscode(void) {
  PyObject *m = Py_InitModule3("_pyoscode", module_methods, module_docstring);
  if (m == NULL)
    return;
  // Load numpy functionality
  import_array();
}
#endif

/* w callback function */
static PyObject *w_callback = NULL;
static PyObject *g_callback = NULL;

static std::complex<double> wfun(double t) {

  std::complex<double> result = 0;
  PyObject *arglist = Py_BuildValue("(d)", t);
  PyObject *py_result = PyObject_CallObject(w_callback, arglist);
  Py_DECREF(arglist);
  // Check whether Python w(t) returned an exception
  if (py_result == NULL)
    return std::complex<double>(NAN, NAN);
  // Check if return value was the correct type (complex)
  if (PyComplex_Check(py_result) || PyFloat_Check(py_result)) {
    result = std::complex<double>(PyComplex_RealAsDouble(py_result),
                                  PyComplex_ImagAsDouble(py_result));
  }
  Py_DECREF(py_result);
  return result;
}

static std::complex<double> gfun(double t) {

  std::complex<double> result = 0;
  PyObject *arglist = Py_BuildValue("(d)", t);
  PyObject *py_result = PyObject_CallObject(g_callback, arglist);
  Py_DECREF(arglist);
  // Check whether Python w(t) returned an exception
  if (py_result == NULL)
    return std::complex<double>(NAN, NAN);
  // Check if return value was the correct type (complex)
  if (PyComplex_Check(py_result) || PyFloat_Check(py_result))
    result = std::complex<double>(PyComplex_RealAsDouble(py_result),
                                  PyComplex_ImagAsDouble(py_result));
  Py_DECREF(py_result);
  return result;
}

struct PyListDeleter {
  void operator()(PyObject* x) { Py_DECREF(x); }
};

using pylist_ptr = std::unique_ptr<PyObject, PyListDeleter>;
pylist_ptr to_pylist(const std::vector<double>& v) {
    pylist_ptr pylist(PyList_New(v.size()));
    for (size_t i = 0; i < v.size(); ++i) {
        PyList_SetItem(pylist.get(), i, PyFloat_FromDouble(v[i]));
    }
    return pylist;
}

pylist_ptr to_pylist(const std::vector<std::complex<double>>& v) {
    pylist_ptr pylist(PyList_New(v.size()));
    for (size_t i = 0; i < v.size(); ++i) {
        Py_complex x_complex{std::real(v[i]), std::imag(v[i])};
        PyList_SetItem(pylist.get(), i, Py_BuildValue("D", &x_complex));
    }
    return pylist;
}

pylist_ptr to_pylist(const std::vector<bool>& v) {
    pylist_ptr pylist(PyList_New(v.size()));
    for (size_t i = 0; i < v.size(); ++i) {
        const bool x = v[i];
        PyList_SetItem(pylist.get(), i, Py_BuildValue("i", &x));
    }
    return pylist;
}

pylist_ptr to_pylist(const std::vector<Eigen::Matrix<std::complex<double>, 7, 1>>& v) {
    pylist_ptr pylist(PyList_New(v.size()));
    for (std::size_t i = 0; i < v.size(); ++i) {
        PyObject *pycts_rep_element = PyList_New(7);
        for (std::size_t j = 0; j <= 6; j++) {
            Py_complex coeff_complex{std::real(v[i](j)), std::imag(v[i](j))};
            PyList_SetItem(pycts_rep_element, j, Py_BuildValue("D", &coeff_complex));
        }
        PyList_SetItem(pylist.get(), i, pycts_rep_element);
    }
    return pylist;
}

/* Function to run the solver when w, g are provided as functions */
static PyObject *_pyoscode_solve_fn(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {

  int order = 3;
  const char *full_output = "";
  double ti, tf, rtol, atol, h0;
  std::complex<double> x0, dx0;
  PyObject *t_evalobj = NULL, *wobj, *gobj;
  // Define keywords
  static const char *kwlist[] = {"w",   "g",           "ti",    "tf",   "x0",
                                 "dx0", "t_eval",      "order", "rtol", "atol",
                                 "h",   "full_output", NULL};
  // Interpret input arguments
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOddDD|Oiddds",
                                   const_cast<char **>(kwlist), &wobj, &gobj,
                                   &ti, &tf, &x0, &dx0, &t_evalobj, &order,
                                   &rtol, &atol, &h0, &full_output))
    return NULL;
  // Set w, g functions
  if (!(PyCallable_Check(wobj) && PyCallable_Check(gobj))) {
    PyErr_SetString(PyExc_TypeError, "Parameter must be callable");
    return NULL;
  }
  // Add reference to new callback
  Py_XINCREF(wobj);
  Py_XINCREF(gobj);
  // Dispose of previous callback
  Py_XDECREF(w_callback);
  Py_XDECREF(g_callback);
  // Remember new callback
  w_callback = wobj;
  g_callback = gobj;

  // If given, read in t_eval, points at which dense output is requested
  PyObject *t_evalarray = NULL;
  if (t_evalobj != NULL) {
    t_evalarray = PyArray_FROM_OTF(t_evalobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  }
  PyArrayObject *t_evalarray_arr = NULL;
  if (t_evalobj != NULL) {
    t_evalarray_arr = reinterpret_cast<PyArrayObject *>(t_evalarray);
  }
  double *t_eval = NULL;
  int t_evalsize = 0;
  if (t_evalobj != NULL) {
    t_eval = (double *)PyArray_DATA(t_evalarray_arr);
    t_evalsize = (int)PyArray_SIZE(t_evalarray_arr);
  }
  std::vector<double> t_evallist;
  t_evallist.resize(t_evalsize);
  int i = 0;
  for (auto it = t_evallist.begin(); it != t_evallist.end(); it++) {
    *it = t_eval[i];
    i++;
  }
  // Checks
  try {
    // Check: dense output points must be inside integration range, and
    // monotonic
    if (t_evalsize != 0) {
      if (std::is_sorted(t_evallist.begin(), t_evallist.end()) == false)
        throw "The points at which dense output is requested are not in "
              "ascending order.";
      else if (h0 > 0 && (t_evallist.front() < ti || t_evallist.back() > tf)) {
        throw "The point at which dense output is requested must lie in the "
              "integration range (between ti, tf).";
      } else if (h0 < 0 &&
                 (t_evallist.front() < tf || t_evallist.back() > ti)) {
        throw "The point at which dense output is requested must lie in the "
              "integration range (between ti, tf).";
      }
    }
    // Check: direction of integration must match initial stepsize sign
    if (((tf - ti) > 0 && h0 < 0) || ((tf - ti) < 0 && h0 > 0))
      throw "Direction of integration ( tf-ti ) in conflict with sign of "
            "initial step (h).";
  } catch (const char *errormsg) {
    PyErr_SetString(PyExc_TypeError, errormsg);
    return (PyObject *)NULL;
  }
  de_system sys = de_system(&wfun, &gfun);
  std::vector<std::complex<double>> sol, dsol;
  std::vector<double> times;
  std::vector<bool> wkbs;
  std::vector<std::complex<double>> x_eval, dx_eval;
  std::vector<Eigen::Matrix<std::complex<double>, 7, 1>> cts_rep;

  if (t_evalobj != NULL) {
    Solution solution(sys, x0, dx0, ti, tf, t_evallist, order, h0, full_output);
    solution.solve(rtol, atol);
    x_eval = solution.dosol_;
    dx_eval = solution.dodsol_;
    sol = solution.sol_;
    dsol = solution.dsol_;
    times = solution.times_;
    wkbs = solution.wkbs_;
    cts_rep = solution.sol_vdm_;
  } else {
    Solution solution(sys, x0, dx0, ti, tf, order, h0, full_output);
    solution.solve(rtol, atol);
    x_eval = solution.dosol_;
    dx_eval = solution.dodsol_;
    sol = solution.sol_;
    dsol = solution.dsol_;
    times = solution.times_;
    wkbs = solution.wkbs_;
    cts_rep = solution.sol_vdm_;
  }
  // Build output values
  auto pyx_eval = to_pylist(x_eval);
  auto pydx_eval = to_pylist(dx_eval);
  auto pycts_rep = to_pylist(cts_rep);
  auto pytimes = to_pylist(times);
  auto pysol = to_pylist(sol);
  auto pydsol = to_pylist(dsol);
  auto pywkbs = to_pylist(wkbs);;  
  PyObject *retdict = Py_BuildValue("{s:O,s:O,s:O,s:O,s:O,s:O,s:O}", "sol", pysol.get(), "dsol",
                          pydsol.get(), "t", pytimes.get(), "types", pywkbs.get(), "x_eval",
                          pyx_eval.get(), "dx_eval", pydx_eval.get(), "cts_rep", pycts_rep.get());
  // Clean up
  // Py_DECREF(pycts_rep_element);
  //    Py_DECREF(t_evalarray);
  //    Py_DECREF(pyx_eval);
  return retdict;
}


/* Function to run the solver when w, g are provided as arrays */
static PyObject *_pyoscode_solve(PyObject *self, PyObject *args,
                                 PyObject *kwargs) {

  int islogw = 0, islogg = 0, order = 3, even = 0, check_grid = 0;
  const char *full_output = "";
  double ti, tf, rtol, atol, h0;
  std::complex<double> x0, dx0;
  PyObject *tsobj, *wsobj, *gsobj, *t_evalobj = NULL;
  // Define keywords
  static const char *kwlist[] = {
      "ts",   "ws",     "gs",          "ti",        "tf",         "x0",
      "dx0",  "t_eval", "logw",        "logg",      "order",      "rtol",
      "atol", "h",      "full_output", "even_grid", "check_grid", NULL};

  // Interpret input arguments.
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OOOddDD|Oiiidddsii", const_cast<char **>(kwlist),
          &tsobj, &wsobj, &gsobj, &ti, &tf, &x0, &dx0, &t_evalobj, &islogw,
          &islogg, &order, &rtol, &atol, &h0, &full_output, &even, &check_grid))
    return NULL;
  // Interpret input objects as numpy arrays
  PyObject *tsarray = PyArray_FROM_OTF(tsobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *wsarray = PyArray_FROM_OTF(wsobj, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *gsarray = PyArray_FROM_OTF(gsobj, NPY_CDOUBLE, NPY_ARRAY_IN_ARRAY);
  PyObject *t_evalarray = NULL;
  if (t_evalobj != NULL) {
    t_evalarray = PyArray_FROM_OTF(t_evalobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  }
  // If that didn't work, throw an exception
  if (tsarray == NULL || wsarray == NULL || gsarray == NULL) {
    Py_XDECREF(tsarray);
    Py_XDECREF(wsarray);
    Py_XDECREF(gsarray);
    return NULL;
  }

  // Get pointers to the data as c++-types
  PyArrayObject *tsarray_arr = reinterpret_cast<PyArrayObject *>(tsarray);
  PyArrayObject *wsarray_arr = reinterpret_cast<PyArrayObject *>(wsarray);
  PyArrayObject *gsarray_arr = reinterpret_cast<PyArrayObject *>(gsarray);
  PyArrayObject *t_evalarray_arr = NULL;
  if (t_evalobj != NULL) {
    t_evalarray_arr = reinterpret_cast<PyArrayObject *>(t_evalarray);
  }
  double *ts = (double *)PyArray_DATA(tsarray_arr);
  std::complex<double> *ws = (std::complex<double> *)PyArray_DATA(wsarray_arr);
  std::complex<double> *gs = (std::complex<double> *)PyArray_DATA(gsarray_arr);
  double *t_eval = NULL;
  int t_evalsize = 0;
  if (t_evalobj != NULL) {
    t_eval = (double *)PyArray_DATA(t_evalarray_arr);
    t_evalsize = (int)PyArray_SIZE(t_evalarray_arr);
  }
  std::vector<double> t_evallist;
  t_evallist.resize(t_evalsize);
  int i = 0;
  for (auto it = t_evallist.begin(); it != t_evallist.end(); it++) {
    *it = t_eval[i];
    i++;
  }

  // Get array sizes
  int tssize = (int)PyArray_SIZE(tsarray_arr);
  int wssize = (int)PyArray_SIZE(wsarray_arr);
  int gssize = (int)PyArray_SIZE(gsarray_arr);

  try {
    // Check 1: same size arrays, large enough arrays
    if (tssize != wssize || wssize != gssize || tssize != gssize)
      throw "The sizes of the input arrays (ts, ws, gs) do not match. Please "
            "supply arrays of the same size.";
    else if (tssize < 2)
      throw "The input arrays (ts, ws, gs) have to have at least size 2.";

    // Check 2: ts must be strictly monotonous
    double dir = ts[1] - ts[0];
    double diff;
    for (int i = 1; i < tssize; i++) {
      if (dir > 0)
        diff = ts[i] - ts[i - 1];
      else
        diff = ts[i - 1] - ts[i];
      if (diff < 0)
        throw "The time array (ts) must be strictly monotonous.";
    }

    // Check 3: integration limits ti, tf must lie inside ts
    if (ti < ts[0] || ti > ts[tssize - 1])
      throw "The start of integration, ti, must lie inside the array ts.";
    else if (tf < ts[0] || tf > ts[tssize - 1])
      throw "The end of integration, tf, must lie inside of the array ts.";

    // Check 4: dense output points must be inside integration range, and
    // monotonic
    if (t_evalsize != 0) {
      if (std::is_sorted(t_evallist.begin(), t_evallist.end()) == false)
        throw "The points at which dense output is requested are not in "
              "ascending order.";
      else if (h0 > 0 && (t_evallist.front() < ti || t_evallist.back() > tf)) {
        throw "The point at which dense output is requested must lie in the "
              "integration range (between ti, tf).";
      } else if (h0 < 0 &&
                 (t_evallist.front() < tf || t_evallist.back() > ti)) {
        throw "The point at which dense output is requested must lie in the "
              "integration range (between ti, tf).";
      }
    }
    // Check 5: direction of integration must match initial stepsize sign
    if (((tf - ti) > 0 && h0 < 0) || ((tf - ti) < 0 && h0 > 0))
      throw "Direction of integration ( tf-ti ) in conflict with sign of "
            "initial step (h).";

  } catch (const char *errormsg) {
    PyErr_SetString(PyExc_TypeError, errormsg);
    // Clean up
    Py_DECREF(tsarray);
    Py_DECREF(wsarray);
    Py_DECREF(gsarray);
    //        if(t_evalobj!=NULL){
    //            Py_DECREF(t_evalarray);
    //        }
    return (PyObject *)NULL;
  }

  // Call the C++ functions to construct system and solve
  de_system sys =
      de_system(ts, ws, gs, ts, tssize, islogw, islogg, even, check_grid);
  if (sys.grid_fine_enough != 1) {
    PyErr_WarnEx(PyExc_RuntimeWarning, "One or more of the arrays provided \
(w, g, logw, logg) may not be fine enough to carry out linear \
interpolation accurately. This may result in not traversing oscillatory \
regions of the solution efficiently and numerical inaccuracies. Please \
consider refining the sampling of the array(s) or switching to a more \
suitable independent variable.",
                 1);
  }
  std::vector<std::complex<double>> sol, dsol;
  std::vector<double> times;
  std::vector<bool> wkbs;
  std::vector<std::complex<double>> x_eval, dx_eval;
  std::vector<Eigen::Matrix<std::complex<double>, 7, 1>> cts_rep;

  if (t_evalobj != NULL) {
    Solution solution(sys, x0, dx0, ti, tf, t_evallist, order, h0, full_output);
    solution.solve(rtol, atol);
    x_eval = solution.dosol_;
    dx_eval = solution.dodsol_;
    sol = solution.sol_;
    dsol = solution.dsol_;
    times = solution.times_;
    wkbs = solution.wkbs_;
    cts_rep = solution.sol_vdm_;
  } else {
    Solution solution(sys, x0, dx0, ti, tf, order, h0, full_output);
    solution.solve(rtol, atol);
    x_eval = solution.dosol_;
    dx_eval = solution.dodsol_;
    sol = solution.sol_;
    dsol = solution.dsol_;
    times = solution.times_;
    wkbs = solution.wkbs_;
    cts_rep = solution.sol_vdm_;
  }
  // Build output values
  auto pyx_eval = to_pylist(x_eval);
  auto pydx_eval = to_pylist(dx_eval);
  auto pycts_rep = to_pylist(cts_rep);
  auto pytimes = to_pylist(times);
  auto pysol = to_pylist(sol);
  auto pydsol = to_pylist(dsol);
  auto pywkbs = to_pylist(wkbs);;  
  PyObject *retdict = Py_BuildValue("{s:O,s:O,s:O,s:O,s:O,s:O,s:O}", "sol", pysol.get(), "dsol",
                          pydsol.get(), "t", pytimes.get(), "types", pywkbs.get(), "x_eval",
                          pyx_eval.get(), "dx_eval", pydx_eval.get(), "cts_rep", pycts_rep.get());
  // Clean up
  Py_DECREF(tsarray);
  Py_DECREF(wsarray);
  Py_DECREF(gsarray);
  // Py_DECREF(pycts_rep_element);
  //    Py_DECREF(t_evalarray);
  //    Py_DECREF(pyx_eval);
  return retdict;
}

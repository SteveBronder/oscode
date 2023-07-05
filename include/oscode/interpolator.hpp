#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <iterator>
#include <algorithm>
#include <vector>

template <typename X = double *, typename Y = std::complex<double> *,
          typename InputIt_x = double *>
struct LinearInterpolator {

public:
  int sign_; // denotes direction of integration
  double xstart, dx;
  X x_;      // array of indep. variable
  Y y_;      // array of dep. variable
  int even_; // Bool, true for evenly spaced grids
  InputIt_x x_lower_bound, x_upper_bound;
  InputIt_x x_lower_it, x_upper_it, x0_it;
  double x_lower, x_upper, h;
  std::complex<double> y_lower, y_upper;

  LinearInterpolator() = default;

  LinearInterpolator(X x, Y y, int even) : even_(even) {
    // Constructor of struct, sets struct members
    if (even_) {
      x_ = x;
      y_ = y;
      xstart = x[0];
      dx = x[1] - x[0];
    } else {
      x_ = x;
      y_ = y;
      // xstart and dx should have defaults in this case
    }
  }

  inline void set_interp_start(InputIt_x x_start) noexcept {
    // Sets iterator pointing to first element of time-array
    x0_it = x_start;
  }

  inline void set_interp_bounds(InputIt_x lower_it,
                                InputIt_x upper_it) noexcept {
    // Sets iterators for lower and upper bounds within which search for
    // nearest neighbours is performed for interpolation.
    x_lower_bound = lower_it;
    x_upper_bound = upper_it;
  }

  inline void update_interp_bounds(bool is_interpolated) noexcept {
    x_lower_bound =
        is_interpolated && even_ && sign_ ? x_upper_it : x_lower_bound;
    x_upper_bound =
        is_interpolated && !even_ && !sign_ ? x_lower_it : x_upper_bound;
  }

  inline void update_interp_bounds_reverse() noexcept {
    x_upper_bound = x_lower_it;
  }

  inline std::complex<double> operator()(double x) noexcept {
    // Does linear interpolation
    if (even_) {
      int i = int((x - xstart) / dx);
      std::complex<double> y0 = y_[i];
      std::complex<double> y1 = y_[i + 1];
      return y0 + (y1 - y0) * (x - xstart - dx * i) / dx;
    } else {
      x_upper_it = std::upper_bound(x_lower_bound, x_upper_bound, x);
      x_lower_it = x_upper_it - 1;
      x_lower = *x_lower_it;
      x_upper = *x_upper_it;
      y_lower = y_[(x_lower_it - x0_it)];
      y_upper = y_[(x_upper_it - x0_it)];
      return (y_lower * (x_upper - x) + y_upper * (x - x_lower)) /
             (x_upper - x_lower);
    }
  }

  inline std::complex<double> expit(double x) noexcept {
    // Does linear interpolation when the input is ln()-d
    if (even_) {
      int i = int((x - xstart) / dx);
      std::complex<double> y0 = y_[i];
      std::complex<double> y1 = y_[i + 1];
      return std::exp(y0 + (y1 - y0) * (x - xstart - dx * i) / dx);
    } else {
      x_upper_it = std::upper_bound(x_lower_bound, x_upper_bound, x);
      x_lower_it = x_upper_it - 1;
      x_lower = *x_lower_it;
      x_upper = *x_upper_it;
      y_lower = y_[(x_lower_it - x0_it)];
      y_upper = y_[(x_upper_it - x0_it)];
      return std::exp((y_lower * (x_upper - x) + y_upper * (x - x_lower)) /
                      (x_upper - x_lower));
    }
  }

  inline int check_grid_fineness(int N) noexcept {

    int success = 1;
    std::complex<double> y0, y1, yprev, ynext;
    double x0, xprev, xnext;
    double err;

    // Check grid fineness here
    for (int i = 2; i < N; i += 2) {
      if (even_) {
        y0 = y_[i - 1];
        y1 = 0.5 * (y_[i] + y_[i - 2]);
        err = std::abs((y1 - y0) / y0);
        if (err > 2e-5) {
          success = 0;
          break;
        }
      } else {
        y0 = y_[i - 1];
        x0 = x_[i - 1];
        xprev = x_[i - 2];
        xnext = x_[i];
        yprev = y_[i - 2];
        ynext = y_[i];
        y1 = (yprev * (xnext - x0) + ynext * (x0 - xprev)) / (xnext - xprev);
        err = std::abs((y1 - y0) / y0);
        if (err > 2e-5) {
          success = 0;
          break;
        }
      }
    }
    return success;
  }
};

#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <iterator>
#include <vector>

template <typename X = double *, typename Y = std::complex<double> *,
          typename InputIt_x = double *>
struct BaseInterpolator {
public:
  std::complex<double> y_lower;
  std::complex<double> y_upper;
  double xstart;
  double dx;
  X x_; // array of indep. variable
  Y y_; // array of dep. variable
  InputIt_x x_lower_bound;
  InputIt_x x_upper_bound;
  InputIt_x x_lower_it;
  InputIt_x x_upper_it;
  InputIt_x x0_it;
  double x_lower;
  double x_upper;
  double h;
  bool sign_{false}; // denotes direction of integration
  bool even_{false}; // Bool, true for evenly spaced grids
  BaseInterpolator(std::initializer_list<int> /* */) {}
  BaseInterpolator(X x, Y y, int even)
      : y_lower(0), y_upper(0),
        xstart(even ? x[0] : std::numeric_limits<double>::quiet_NaN()),
        dx(even ? x[1] - x[0] : std::numeric_limits<double>::quiet_NaN()),
        x_(x), y_(y), x_lower_bound(nullptr), x_upper_bound(nullptr),
        x_lower_it(nullptr), x_upper_it(nullptr), x0_it(nullptr), x_lower(0),
        x_upper(0), h(0), sign_(false), even_(even) {}

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
  inline auto sign() const noexcept { return sign_; }
};

template <typename X = double *, typename Y = std::complex<double> *,
          typename InputIt_x = double *>
struct LinearInterpolator : public BaseInterpolator<X, Y, InputIt_x> {

public:
  LinearInterpolator(X x, Y y, int even)
      : BaseInterpolator<X, Y, InputIt_x>(x, y, even) {}
  inline std::complex<double> operator()(double x) noexcept {
    // Does linear interpolation
    if (this->even_) {
      int i = int((x - this->xstart) / this->dx);
      std::complex<double> y0 = this->y_[i];
      std::complex<double> y1 = this->y_[i + 1];
      return y0 + (y1 - y0) * (x - this->xstart - this->dx * i) / this->dx;
    } else {
      this->x_upper_it =
          std::upper_bound(this->x_lower_bound, this->x_upper_bound, x);
      this->x_lower_it = this->x_upper_it - 1;
      this->x_lower = *(this->x_lower_it);
      this->x_upper = *(this->x_upper_it);
      this->y_lower = this->y_[(this->x_lower_it - this->x0_it)];
      this->y_upper = this->y_[(this->x_upper_it - this->x0_it)];
      return (this->y_lower * (this->x_upper - x) +
              this->y_upper * (x - this->x_lower)) /
             (this->x_upper - this->x_lower);
    }
  }
};

template <typename X = double *, typename Y = std::complex<double> *,
          typename InputIt_x = double *>
struct LogLinearInterpolator : public BaseInterpolator<X, Y, InputIt_x> {

public:
  LogLinearInterpolator(X x, Y y, int even)
      : BaseInterpolator<X, Y, InputIt_x>(x, y, even) {}
  inline std::complex<double> operator()(double x) noexcept {
    // Does linear interpolation when the input is ln()-d
    if (this->even_) {
      int i = int((x - this->xstart) / this->dx);
      std::complex<double> y0 = this->y_[i];
      std::complex<double> y1 = this->y_[i + 1];
      return std::exp(y0 +
                      (y1 - y0) * (x - this->xstart - this->dx * i) / this->dx);
    } else {
      this->x_upper_it =
          std::upper_bound(this->x_lower_bound, this->x_upper_bound, x);
      this->x_lower_it = this->x_upper_it - 1;
      this->x_lower = *(this->x_lower_it);
      this->x_upper = *(this->x_upper_it);
      this->y_lower = this->y_[(this->x_lower_it - this->x0_it)];
      this->y_upper = this->y_[(this->x_upper_it - this->x0_it)];
      return std::exp((this->y_lower * (this->x_upper - x) +
                       this->y_upper * (x - this->x_lower)) /
                      (this->x_upper - this->x_lower));
    }
  }
};

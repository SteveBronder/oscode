#pragma once
#include <oscode/interpolator.hpp>
#include <vector>
/** */
class de_system {
private:
  int even_;

public:
  template <typename X, typename Y, typename Z, typename X_it>
  de_system(X &ts, Y &ws, Z &gs, X_it x_it, int size, bool isglogw = false,
            bool islogg = false, int even = 0, int check_grid = 0);
  de_system(std::complex<double> (*)(double, void *),
            std::complex<double> (*)(double, void *), void *);
  de_system(std::complex<double> (*)(double), std::complex<double> (*)(double));
  de_system();
  std::function<std::complex<double>(double)> w_;
  std::function<std::complex<double>(double)> g_;
  LinearInterpolator<> Winterp_;
  LinearInterpolator<> Ginterp_;
  bool islogg_, islogw_;
  bool grid_fine_enough = 1;
  bool is_interpolated_;
};

/** Default contructor */
de_system::de_system() {}

/** Constructor for the case of the user having defined the frequency and
 * damping terms as sequences
 */
template <typename X, typename Y, typename Z, typename X_it>
de_system::de_system(X &ts, Y &ws, Z &gs, X_it x_it, int size, bool islogw,
                     bool islogg, int even, int check_grid) {

  is_interpolated_ = 1;
  even_ = even;
  islogg_ = islogg;
  islogw_ = islogw;

  /** Set up interpolation on the supplied frequency and damping arrays */
  LinearInterpolator<X, Y, X_it> winterp(ts, ws, even_);
  LinearInterpolator<X, Z, X_it> ginterp(ts, gs, even_);

  Winterp_ = winterp;
  Ginterp_ = ginterp;

  if (!even_) {
    Winterp_.set_interp_start(x_it);
    Ginterp_.set_interp_start(x_it);
    Winterp_.set_interp_bounds(ts, ts + size - 1);
    Ginterp_.set_interp_bounds(ts, ts + size - 1);
  }

  /** Check if supplied grids are sampled finely enough for the purposes of
   * linear interpolation
   */
  if (check_grid) {
    int w_is_fine = Winterp_.check_grid_fineness(size);
    int g_is_fine = Ginterp_.check_grid_fineness(size);
    if (w_is_fine && g_is_fine)
      grid_fine_enough = 1;
    else
      grid_fine_enough = 0;
  }

  /** Bind result of interpolation to a function, this will be called by the
   * routines taking RK and WKB steps
   */
  if (islogw)
    w_ = std::bind(&LinearInterpolator<X, Y, X_it>::expit, Winterp_,
                   std::placeholders::_1);
  else
    w_ = std::bind(&LinearInterpolator<X, Y, X_it>::operator(), Winterp_,
                   std::placeholders::_1);
  if (islogg)
    g_ = std::bind(&LinearInterpolator<X, Z, X_it>::expit, Ginterp_,
                   std::placeholders::_1);
  else
    g_ = std::bind(&LinearInterpolator<X, Z, X_it>::operator(), Ginterp_,
                   std::placeholders::_1);
}

/** Constructor for the case when the frequency and damping terms have been
 * defined as functions
 */
de_system::de_system(std::complex<double> (*W)(double, void *),
                     std::complex<double> (*G)(double, void *), void *p) {

  is_interpolated_ = 0;
  w_ = [W, p](double x) { return W(x, p); };
  g_ = [G, p](double x) { return G(x, p); };
};

/** Constructor for the case when the frequency and damping terms have been
 * defined as functions (and there are no additional parameters that the
 * function might need)
 */
de_system::de_system(std::complex<double> (*W)(double),
                     std::complex<double> (*G)(double)) {

  is_interpolated_ = 0;
  w_ = W;
  g_ = G;
};

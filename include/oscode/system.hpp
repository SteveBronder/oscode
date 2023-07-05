#pragma once
#include <functional>
#include <oscode/interpolator.hpp>
#include <vector>

/** */
template <typename WInterp, typename GInterp> class de_system {

public:
  template <typename X, typename Y, typename Z, typename X_it,
            typename WWInterp, typename GGInterp>
  de_system(X &ts, Y &ws, Z &gs, X_it x_it, int size, WWInterp &&w_interp,
            GGInterp &&g_interp, int even, int check_grid)
      : even_(even), Winterp_(std::forward<WWInterp>(w_interp)),
        Ginterp_(std::forward<GGInterp>(g_interp)), is_interpolated_(true) {
    /** Set the interpolation bounds */
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
      if (w_is_fine && g_is_fine) {
        grid_fine_enough = true;
      } else {
        grid_fine_enough = false;
      }
    }
  }

  /** Constructor for the case when the frequency and damping terms have been
   * defined as functions (and there are no additional parameters that the
   * function might need)
   */
  template <typename WWInterp, typename GGInterp>
  de_system(WWInterp &&w_interp, GGInterp &&g_interp)
      : Winterp_(std::forward<WWInterp>(w_interp)),
        Ginterp_(std::forward<GGInterp>(g_interp)), grid_fine_enough(true),
        is_interpolated_(false) {}
  de_system(){};

private:
  int even_;

public:
  WInterp Winterp_;
  GInterp Ginterp_;
  bool grid_fine_enough{true};
  bool is_interpolated_{false};
};

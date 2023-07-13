#pragma once
#include <functional>
#include <oscode/interpolator.hpp>
/** */
template <typename WInterp, typename GInterp> class de_system {
private:
  int even_;

public:
  /** Default contructor */
  de_system() {}

  /** Constructor for the case of the user having defined the frequency and
   * damping terms as sequences
   */
  template <typename X, typename Y, typename Z, typename X_it,
            typename WWInterp, typename GGInterp>
  de_system(X &ts, Y &ws, Z &gs, X_it x_it, int size, WWInterp &&w_interp,
            GGInterp &&g_interp, int even, int check_grid) : 
            even_(even),
            w_func_(std::forward<WWInterp>(w_interp)), 
            g_func_(std::forward<GGInterp>(g_interp)) {
 
    if (even_ == 0) {
      w_func_.set_interp_start(x_it);
      g_func_.set_interp_start(x_it);
      w_func_.set_interp_bounds(ts, ts + size - 1);
      g_func_.set_interp_bounds(ts, ts + size - 1);
    }

    /** Check if supplied grids are sampled finely enough for the purposes of
     * linear interpolation
     */
    if (check_grid == 1) {
      grid_fine_enough = w_func_.check_grid_fineness(size) && g_func_.check_grid_fineness(size);
    }
  }
  template <typename WWInterp, typename GGInterp>
  de_system(WWInterp &&w_interp, GGInterp &&g_interp) : 
            even_(false),
              w_func_(std::forward<WWInterp>(w_interp)), 
            g_func_(std::forward<GGInterp>(g_interp)) {}

  WInterp w_func_;
  GInterp g_func_;
  bool grid_fine_enough = 1;
  static constexpr bool is_interpolated =
      WInterp::is_interpolated || GInterp::is_interpolated;
};

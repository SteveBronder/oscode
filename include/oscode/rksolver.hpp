#pragma once
#include <Eigen/Dense>
#include <complex>
#include <functional>
#include <list>
#include <oscode/system.hpp>
#include <vector>

/** Class that defines a 4 and 5th order Runge-Kutta method.
 *
 * This is a "bespoke"
 * Runge-Kutta formula based on the nodes used in 5th and 6th order
 * Gauss-Lobatto integration, as detailed in [1].
 *
 * [1] Agocs, F. J., et al. “Efficient Method for Solving Highly Oscillatory
 * Ordinary Differential Equations with Applications to Physical Systems.”
 * Physical Review Research, vol. 2, no. 1, 2020,
 * doi:10.1103/physrevresearch.2.013030.
 */
class RKSolver {
public:
  /** Defines the ODE */
  de_system *de_sys_;
  // grid of ws, gs_
  /** 6 values of the frequency term per step, evaluated at the nodes of 6th
   * order Gauss-Lobatto quadrature
   */
  Eigen::Matrix<std::complex<double>, 6, 1> ws_;
  /** 6 values of the friction term per step, evaluated at the nodes of 6th
   * order Gauss-Lobatto quadrature
   */
  Eigen::Matrix<std::complex<double>, 6, 1> gs_;
  /** 5 values of the frequency term per step, evaluated at the nodes of 5th
   * order Gauss-Lobatto quadrature
   */
  Eigen::Matrix<std::complex<double>, 5, 1> ws5_;
  /** 5 values of the friction term per step, evaluated at the nodes of 5th
   * order Gauss-Lobatto quadrature
   */
  Eigen::Matrix<std::complex<double>, 5, 1> gs5_;
  Eigen::Matrix<std::complex<double>, 2, 6> k5_;
  Eigen::Matrix<std::complex<double>, 2, 7> k_dense_;
  /* clang-format off */
  Eigen::Matrix<double, 7, 4> P_dense_{
    {1., -2.48711376,  2.42525041, -0.82538093},
    {0.,  0.,          0.,          0.}, 
    {0.,  3.78546138, -5.54469086,  2.26578746}, 
    {0., -0.27734213,  0.74788587, -0.42224334}, 
    {0., -2.94848704,  7.41087391, -4.08391191}, 
    {0.,  0.50817346, -1.20070313,  0.64644062},
    {0.,  1.4193081,  -3.8386162,   2.4193081}};
  /* clang-format on */
  // Experimental continuous representation of solution
  Eigen::Matrix<std::complex<double>, 7, 1> x_vdm_;

private:
  // Frequency and friction term
  std::function<std::complex<double>(double)> w_;
  std::function<std::complex<double>(double)> g_;

  // Butcher tablaus
  /* clang-format off */
  Eigen::Matrix<double, 5, 5> butcher_a5_{
   { 0.1174723380352676535740, 0,                        0,                        0,                        0},
   {-0.1862479800651504276304, 0.5436322218248278794734, 0,                        0,                        0},
   {-0.6064303885508280518989, 1,                        0.2490461467911506000559, 0,                        0},
   { 2.899356540015731406420, -4.368525611566240669139,  2.133806714786316899991,  0.2178900187289247091542, 0},
   {18.67996349995727204273, -28.85057783973131956546,  10.72053408420926869789,   1.414741756508049078612, -0.9646615009432702537787}
  };
  Eigen::Matrix<double, 3, 3> butcher_a4_{
   { 0.172673164646011428100, -1.568317088384971429762,  -8.769507466172720011410},
   { 0,                        2.395643923738960001662,  10.97821961869480000808},
   { 0,                        0,                        -1.208712152522079996671}
  };
  /* clang-format on */
  Eigen::Matrix<double, 6, 1> butcher_b5{
      {0.1127557227351729739820, 0, 0.5065579732655351768382,
       0.04830040376995117617928, 0.3784749562978469803166,
       -0.04608905606850630731611}},
      butcher_c5{{0, 0.117472338035267653574, 0.357384241759677451843,
                  0.642615758240322548157, 0.882527661964732346426, 1}},
      dense_b5{{0.2089555395, 0., 0.7699501023, 0.009438629906, -0.003746982422,
                0.01540271068}};
  Eigen::Matrix<double, 4, 1> butcher_b4_{
      {-0.08333333333333333333558, 0.5833333333333333333357,
       0.5833333333333333333356, -0.08333333333333333333558}},
      butcher_c4{{0, 0.172673164646011428100, 0.827326835353988571900, 1}};

  // Current values of w, g
  std::complex<double> wi_, gi_;

public:
  std::tuple<std::function<std::complex<double>(double)>,
             std::function<std::complex<double>(double)>>
      wi_gi_funcs;
  auto setup_wi_gi_funcs(de_system *de_sys) {
    std::tuple<std::function<std::complex<double>(double)>,
               std::function<std::complex<double>(double)>>
        funcs;

    if (de_sys_->is_interpolated_) {
      if (de_sys_->islogw_) {
        std::get<0>(funcs) = [de_sys](double x) {
          return de_sys->Winterp_.expit(x);
        };
      } else {
        std::get<0>(funcs) = [de_sys](double x) { return de_sys->Winterp_(x); };
      }
      if (de_sys_->islogg_) {
        std::get<1>(funcs) = [de_sys](double x) {
          return de_sys->Ginterp_.expit(x);
        };
      } else {
        std::get<1>(funcs) = [de_sys](double x) { return de_sys->Ginterp_(x); };
      }
    } else {
      std::get<0>(funcs) = [ww = this->w_](double x) { return ww(x); };
      std::get<1>(funcs) = [gg = this->g_](double x) { return gg(x); };
    }
    return funcs;
  }
  /** Callable that gives the frequency term in the ODE at a given time */
  // std::function<std::complex<double>(double)> w;

  // constructors
  /** Default constructor. */
  RKSolver() {}
  /** Constructor for the RKSolver class. It sets up the Butcher tableaus for
   * the two Runge-Kutta methods (4th and 5th order) used.
   *
   * @param de_sys[in] the system of first-order equations defining the
   * second-order ODE to solve.
   */
  RKSolver(de_system &de_sys)
      : de_sys_(&de_sys),
        w_(!de_sys_->is_interpolated_ ? de_sys_->w_ : nullptr),
        g_(!de_sys_->is_interpolated_ ? de_sys_->g_ : nullptr),
        wi_gi_funcs(setup_wi_gi_funcs(de_sys_)) {}

/** Computes a single Runge-Kutta type step, and returns the solution and its
 * local error estimate.
 *
 *
 */
Eigen::Matrix<std::complex<double>, 2, 2>
step(std::complex<double> x0, std::complex<double> dx0, double t0,
               double h) {

  // TODO: resizing of ws5_, gs5_, insertion
  Eigen::Matrix<std::complex<double>, 2, 4> k4;
  Eigen::Matrix<std::complex<double>, 2, 2> result =
      Eigen::Matrix<std::complex<double>, 2, 2>::Zero();
  //    std::cout << "Set up RK step" << std::endl;
  Eigen::Matrix<std::complex<double>, 2, 1> y0{{x0, dx0}};
  k5_.col(0) = h * f(t0, y0);
  //    std::cout << "Asked for f" << std::endl;
  ws_(0) = wi_;
  gs_(0) = gi_;
  Eigen::Matrix<std::complex<double>, 2, 1> y;
  for (int s = 1; s < 6; s++) {
    y = y0;
    for (int i = 0; i < s; i++) {
      y += butcher_a5_(s - 1, i) * k5_.col(i);
    }
    k5_.col(s) = h * f(t0 + butcher_c5(s) * h, y);
    ws_(s) = wi_;
    gs_(s) = gi_;
  }
  k4.col(0) = k5_.col(0);
  ws5_(0) = ws_(0);
  gs5_(0) = gs_(0);
  for (int s = 1; s < 4; s++) {
    y = y0;
    for (int i = 0; i <= (s - 1); i++) {
      y += butcher_a4_(i, s - 1) * k4.col(i);
    }
    k4.col(s) = h * f(t0 + butcher_c4(s) * h, y);
    ws5_(s) = wi_;
    gs5_(s) = gi_;
  }
  result.col(0) += k5_ * butcher_b5;
  result.col(1) = result.col(0) - (k4 * butcher_b4_);
  result.col(0) += y0;
  // result << y5, delta;
  //  Add in missing w, g at t+h/2
  ws5_(4) = ws5_(3);
  ws5_(3) = ws5_(2);
  gs5_(4) = gs5_(3);
  gs5_(3) = gs5_(2);
  const auto next_step = t0 + h / 2;
  ws5_(2) = std::get<0>(this->wi_gi_funcs)(next_step);
  gs5_(2) = std::get<1>(this->wi_gi_funcs)(next_step);

  // Fill up k_dense matrix for dense output
  k_dense_.block<2, 6>(0, 0) = k5_.block<2, 6>(0, 0);
  k_dense_.col(6) = h * f(t0 + h, result.col(0));

  // Experimental continuous output
  Eigen::Matrix<std::complex<double>, 1, 4> Q_dense =
      (k_dense_ * P_dense_).row(0);
  x_vdm_(0) = x0;
  x_vdm_.tail(6) << Q_dense(0), Q_dense(1), Q_dense(2), Q_dense(3), 0.0, 0.0;

  return result;
}


/** Turns the second-order ODE into a system of first-order ODEs as follows:
 *
 * \f[ y = [x, \dot{x}], \f]
 * \f[ \dot{y[0]} = y[1], \f]
 * \f[ \dot{y[1]} = -\omega^2(t)y[0] -2\gamma(t)y[1]. \f]
 *
 * @param t[in] time \f$ t \f$
 * @param y[in] vector of unknowns \f$ y = [x, \dot{x}] \f$
 * @returns a vector of the derivative of \f$ y \f$
 */
template <typename Mat>
Eigen::Matrix<std::complex<double>, 1, 2> f(double t, const Mat &y) {
  wi_ = std::get<0>(this->wi_gi_funcs)(t);
  gi_ = std::get<1>(this->wi_gi_funcs)(t);
  return Eigen::Matrix<std::complex<double>, 1, 2>{
      {y[1], -wi_ * wi_ * y[0] - 2.0 * gi_ * y[1]}};
}
/** Gives dense output at a single point during the step "for free", i.e. at no
 * extra evaluations of \f$ \omega(t), \gamma(t) \f$. This solution is roughly
 * mid-way through the step at $\sigma \sim 0.59 \f$, where \f$ \sigma = 0 \f$
 * corresponds to the start, \f$ \sigma = 1 \f$ to the end of the step.
 */
Eigen::Matrix<std::complex<double>, 1, 2>
dense_point(std::complex<double> x, std::complex<double> dx,
                      const Eigen::Matrix<std::complex<double>, 6, 2> &k5) {

  Eigen::Matrix<std::complex<double>, 1, 2> ydense{{x, dx}};
  for (int j = 0; j <= 5; j++) {
    ydense += 0.5866586817 * dense_b5(j) * k5.row(j);
  }
  return ydense;
}
/** Calculated dense output at a given set of points during a step after a
 * successful Runge-Kutta type step.
 */
void dense_step(double t0, double h0, std::complex<double> y0,
                          std::complex<double> dy0,
                          const std::vector<double> &dots,
                          std::vector<std::complex<double>> &doxs,
                          std::vector<std::complex<double>> &dodxs) {
  const auto docount = dots.size();
  Eigen::Matrix<double, -1, 4> R_dense(docount, 4);
  R_dense.col(0).array() = (Eigen::Map<const Eigen::Array<double, -1, 1>>(dots.data(), dots.size()) - t0) / h0;
  R_dense.col(1).array() = R_dense.col(0).array().square();
  R_dense.col(2).array() = R_dense.col(1).array() * R_dense.col(0).array(); 
  R_dense.col(3).array() = R_dense.col(2).array() * R_dense.col(0).array();
  // Q_dense is used once and could be inline on the line for Y_dense
  Eigen::Matrix<std::complex<double>, -1, 2> Y_dense = R_dense * (k_dense_ * P_dense_).transpose();
  Eigen::Map<Eigen::Array<std::complex<double>, -1, 1>> doxs_vec(doxs.data(), doxs.size());
  Eigen::Map<Eigen::Array<std::complex<double>, -1, 1>> dodxs_vec(dodxs.data(), dodxs.size());
  doxs_vec = y0 + Y_dense.col(0).array();
  dodxs_vec = dy0 + Y_dense.col(1).array();
}
};








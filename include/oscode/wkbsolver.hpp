#pragma once
#include <Eigen/Dense>
#include <complex>
#include <iomanip>
#include <list>
#include <oscode/macros.hpp>
#include <oscode/system.hpp>
#include <vector>

  struct series {
    // WKB series
     Eigen::Matrix<std::complex<double>, 4, 1> val_;
    // Error in WKB series
     Eigen::Matrix<std::complex<double>, 4, 1> error_;
  } ;

/** Class to carry out WKB steps of varying orders.  */
class WKBSolver {
protected:
  template <long int N>
  using eigen_vec_c = Eigen::Matrix<std::complex<double>, N, 1>;
  template <long int N> using eigen_vec_d = Eigen::Matrix<double, N, 1>;
  // WKB series and derivatives (order dependent)
  virtual eigen_vec_c<4> dds(const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const std::complex<double>& d4_w1,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs) {
      return {std::complex<double>(0.0, 1.0) * d1.coeffRef(0, 0), 0.0, 0.0, 0.0};
  }
  virtual eigen_vec_c<4> dsi(const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs) {
      return {std::complex<double>(0.0, 1.0) * ws.coeffRef(0), 0.0, 0.0, 0.0};
    }
  virtual eigen_vec_c<4> dsf(const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs) {
    return {std::complex<double>(0.0, 1.0) * ws.coeffRef(5), 0.0, 0.0, 0.0};
  }
  // Gauss-Lobatto integration
  inline Eigen::Matrix<std::complex<double>, 2, 1>
  integrate(const double h, const Eigen::Matrix<std::complex<double>, 6, 1> &integrand6,
            const Eigen::Matrix<std::complex<double>, 5, 1> &integrand5) const {
    std::complex<double> x6 = h / 2.0 * glws6_.dot(integrand6);
    return {x6, x6 - (h / 2.0 * glws5_.dot(integrand5))};
  }
  virtual series s(const double h, const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 5, 1>& ws5,
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs,
    const Eigen::Matrix<std::complex<double>, 5, 1>& gs5) {
    eigen_vec_c<2> s0 = std::complex<double>(0, 1) * integrate(h, ws, ws5);
    return series{{s0(0), 0.0, 0.0, 0.0}, {s0(1), 0.0, 0.0, 0.0}};
  }

  // Gauss-Lobatto n=6, 5 weights
  // Set Gauss-Lobatto weights
  eigen_vec_d<6> glws6_{{1.0 / 15.0, 0.3784749562978469803166,
                         0.5548583770354863530167, 0.5548583770354863530167,
                         0.3784749562978469803166, 1.0 / 15.0}};
  eigen_vec_d<5> glws5_{
      {1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0}};
  // weights for derivatives
  /** clang-format off */
    Eigen::Matrix<double, 6, 6> d1_w_
  {{-15.0000000048537,   20.2828318761850,     -8.07237453994912,      4.48936929577350,     -2.69982662677053,     0.999999999614819},
    { -3.57272991033049,   0.298532922755350e-7, 5.04685352597795,     -2.30565629452303,      1.30709499910514,    -0.475562350082855},
    {  0.969902096162109, -3.44251390568294,    -0.781532641131861e-10, 3.50592393061265,     -1.57271334190619,     0.539401220892526},
    { -0.539401220892533,  1.57271334190621,    -3.50592393061268,      0.782075077478921e-10, 3.44251390568290,    -0.969902096162095},
    {  0.475562350082834, -1.30709499910509,     2.30565629452296,     -5.04685352597787,     -0.298533681980831e-7, 3.57272991033053},
    { -0.999999999614890,  2.69982662677075,    -4.48936929577383,      8.07237453994954,    -20.2828318761854,     15.0000000048538}
  };
  Eigen::Matrix<double, 6, 6> d2_w_ {
    {140.000000016641, -263.163968874741,   196.996471291466, -120.708905753218,   74.8764032980854, -27.9999999782328},
    { 60.8267436465252, -96.4575130414144,   42.0725563562029,  -8.78105375967028,  3.41699471496020, -1.07772791660362},
    { -5.42778322782674, 28.6981500482483,  -43.5424868874619,  24.5830052399403,  -5.98965265951073,  1.67876748661075},
    {  1.67876748661071, -5.98965265951067,  24.5830052399402, -43.5424868874617,  28.6981500482481,  -5.42778322782664},
    { -1.07772791660381,  3.41699471496078,  -8.78105375967105, 42.0725563562040, -96.4575130414154,  60.8267436465256},
    {-27.9999999782335,  74.8764032980873, -120.708905753221,  196.996471291469, -263.163968874744,  140.000000016642},
  };
  Eigen::Matrix<double, 6, 6> d3_w_ {
    {-840.000000234078, 1798.12714381468, -1736.74461287884, 1322.01528240287, -879.397812956524, 335.999999851893},
    {-519.5390172614027, 1067.7171515309801, -934.3207515371753, 617.0298708048756, -364.83838902593686, 133.95113548865842},
    {-81.13326349151461, 90.82824880172825, 81.1176254157912, -199.41150112141258, 171.22231114098616, -62.62342074557803},
    {62.62342074557783, -171.2223111409866, 199.41150112141344, -81.11762541579242, -90.82824880172696, 81.13326349151436},
    {-133.95113548865962, 364.8383890259409, -617.0298708048813, 934.3207515371842, -1067.717151530989, 519.5390172614059},
    {-335.999999851897, 879.397812956534, -1322.01528240289, 1736.74461287886, -1798.12714381470, 840.000000234086},
  };
  /** clang-format on */
  eigen_vec_d<7> d4w1_w_{{3024.00000383582, -6923.06197480357, 7684.77676018742,
                          0.0, -6855.31809730784, 5085.60330881706,
                          -2016.00000072890}};
      // Above is into matrices
  eigen_vec_d<6>  d3g1_w_{{-840.000000234078, 1798.12714381468, -1736.74461287884, 1322.01528240287, -879.397812956524, 335.999999851893}};
  Eigen::Matrix<double, 3, 5> d1_w_5_{
  {-2.48198050935042, 0.560400997591235e-8, 3.49148624058567, -1.52752523062733, 0.518019493788063},
  {0.750000000213852, -2.67316915534181, 0.360673032443906e-10, 2.67316915534181, -0.750000000213853},
  {-0.518019493788065, 1.52752523062733, -3.49148624058568, -0.560400043118500e-8, 2.48198050935041}
  };    
  
  // grid of ws, gs
  eigen_vec_c<7> ws7_;
  eigen_vec_c<6> ws_, gs_;
  eigen_vec_c<5> ws5_, gs5_;
  // derivatives
  Eigen::Matrix<std::complex<double>, 6, 2> d1_;
  Eigen::Matrix<std::complex<double>, 6, 2> d2_;
  Eigen::Matrix<std::complex<double>, 6, 1> d3_;
  Eigen::Matrix<std::complex<double>, 3, 1> d1_5_;
  std::complex<double> d3g1_, d4w1_;
  std::complex<double> d1w2_5_, d1w3_5_, d1w4_5_;
  eigen_vec_c<5> dws5_;

  // WKB solutions and their derivatives
  std::complex<double> fp_, fm_, dfpi_, dfmi_, dfpf_, dfmf_, ddfp_, ddfm_, ap_,
      am_, bp_, bm_;
  std::complex<double> x_;
  std::complex<double> dx_;
  std::complex<double> ddx_;
  // order
  int order_;
  // error estimate on step
  std::complex<double> err_fp_;
  std::complex<double> err_fm_;
  std::complex<double> err_dfp_;
  std::complex<double> err_dfm_;
  // dense output
  std::vector<std::complex<double>> doxs;
  std::vector<std::complex<double>> dodxs;
  std::vector<std::complex<double>> dows;
    eigen_vec_c<4> dense_ds_;
  eigen_vec_c<4> dense_ds_i;
  std::complex<double> dense_ap_;
  std::complex<double> dense_am_;
  std::complex<double> dense_bp_;
  std::complex<double> dense_bm_;
    // Experimental continuous representation of solution

  // experimental dense output + quadrature
  // Set polynomial Gauss--Lobatto coefficients for dense output + quadrature
  // clang-format off
  Eigen::Matrix<double, 6, 6> integ_vandermonde_{
    {0.06250000000, -0.1946486424, 0.6321486424, 0.6321486424, -0.1946486424, 0.06250000000}, 
   {-0.03125000000,  0.1272121350,-1.108132527,  1.108132527,  -0.1272121350, 0.03125000000},
   {-0.2916666667,   0.8623909801,-0.5707243134,-0.5707243134,  0.8623909801,-0.2916666667}, 
    {0.2187500000,  -0.8454202132, 1.500687022, -1.500687022,   0.8454202132,-0.2187500000}, 
    {0.2625000000,  -0.4785048596, 0.2160048596, 0.2160048596, -0.4785048596, 0.2625000000},
   {-0.2187500000,   0.5212094304,-0.6310805056, 0.6310805056, -0.5212094304, 0.2187500000}
  }; 
  Eigen::Matrix<double, 6, 6> interp_vandermonde_{
    {0.06250000000, -0.1946486424, 0.6321486424, 0.6321486424, -0.1946486424, 0.06250000000},
   {-0.06250000000,  0.2544242701,-2.216265054,  2.216265054,  -0.2544242701, 0.06250000000},
   {-0.8750000000,   2.587172940, -1.712172940, -1.712172940,   2.587172940, -0.8750000000}, 
    {0.8750000000,  -3.381680853,  6.002748088, -6.002748088,   3.381680853, -0.8750000000}, 
    {1.312500000,   -2.392524298,  1.080024298,  1.080024298,  -2.392524298,  1.312500000}, 
   {-1.312500000,    3.127256583, -3.786483034,  3.786483034,  -3.127256583,  1.312500000}};
  // clang-format on
public:
  Eigen::Matrix<std::complex<double>, 7, 1> x_vdm;
  // constructor
  WKBSolver(){};
  WKBSolver(de_system &de_sys, int order) : order_(order){};
  virtual ~WKBSolver() {}
    Eigen::Matrix<std::complex<double>, 3, 2>
  step(bool dense_output, std::complex<double> x0, std::complex<double> dx0, double t0, double h0,
       const Eigen::Matrix<std::complex<double>, 6, 1> &ws,
       const Eigen::Matrix<std::complex<double>, 6, 1> &gs,
       const Eigen::Matrix<std::complex<double>, 5, 1> &ws5,
       const Eigen::Matrix<std::complex<double>, 5, 1> &gs5) {
       if (dense_output) {
        return step_internal<true>(x0, dx0, t0, h0, ws, gs, ws5, gs5);
       } else {
        return step_internal<false>(x0, dx0, t0, h0, ws, gs, ws5, gs5);
       }
  }
  /** Computes a WKB step of a given order and returns the solution and its
   * local error estimate.
   *
   *
   * @param x0[in] value of the solution \f$x(t)\f$ at the start of the step
   * @param dx0[in] value of the derivative of the solution \f$\frac{dx}{dt}\f$
   * at the start of the step
   * @param t0[in] value of independent variable (time) at the start of the step
   * @param h0[in] size of the step (solution will be given at t+t0)
   * @param ws[in] vector of 6 evaulations of the frequency term at the
   * Gauss-Lobatto nodes, this is necessary for Gauss-Lobatto integration, which
   * in turn is needed to calculate the WKB series
   * @param gs[in] vector of 6 evaluations of the friction term
   * @param ws5[in] vector of 5 evaluations of the friction term at the nodes of
   * 5th order Gauss-Lobatto quadrature, this is needed to compute the error on
   * Gauss-Lobatto quadrature, and in turn the WKB series
   * @param gs5[in] vector of 5 evaluations of the friction term
   *
   * @returns a matrix, whose rows are:
   * - \f$ x, \dot{x}\f$ at t0+h0 as an nth order WKB estimate
   * - \f$ \Delta_{\mathrm{trunc}}x, \Delta_{\mathrm{trunc}}\dot{x}\f$ at t0+h0,
   * defined as the difference between an nth and (n-1)th order WKB estimate
   * - \f$ \Delta_{\mathrm{int}}x, \Delta_{\mathrm{int}}\dot{x}\f$ at t0+h0,
   *   defined as the local error coming from those terms in the WKB series that
   *   involved numerical integrals
   */
  template <bool dense_output>
  Eigen::Matrix<std::complex<double>, 3, 2>
  step_internal(std::complex<double> x0, std::complex<double> dx0, double t0, double h0,
       const Eigen::Matrix<std::complex<double>, 6, 1> &ws,
       const Eigen::Matrix<std::complex<double>, 6, 1> &gs,
       const Eigen::Matrix<std::complex<double>, 5, 1> &ws5,
       const Eigen::Matrix<std::complex<double>, 5, 1> &gs5) {

    // Set grid of ws, gs:
    Eigen::Matrix<std::complex<double>, 6, 2> ws_gs;
    ws_gs.col(0) = ws;
    ws_gs.col(1) = gs;
    ws_ = ws;
    gs_ = gs;
    ws5_ = ws5;
    gs5_ = gs5;
    ws7_ = Eigen::Matrix<std::complex<double>, 7, 1>{
        {ws_(0), ws_(1), ws_(2), ws5_(2), ws_(3), ws_(4), ws_(5)}};
    // Set i.c.
    x_ = x0;
    dx_ = dx0;
    ddx_ = -std::pow(ws_(0), 2) * x_ - 2.0 * gs_(0) * dx_;
    // step and stepsize
    const double h = h0;
    if constexpr (OSCODE_DEBUG) {
      std::cout << "ws_: \n" << ws_ << std::endl; 
    }
    d1_ = ((d1_w_ * ws_gs).array() / h).matrix(); 
    const auto h_sq = h * h;
    d2_ = ((d2_w_ * ws_gs).array() / h_sq).matrix(); 
    const auto h_cube = h_sq * h;
    d3_ = ((d3_w_ * ws_).array() / h_cube).matrix().transpose(); 
    d4w1_ = d4w1_w_.dot(ws7_) / (h_cube * h);
    d3g1_ = d3g1_w_.dot(gs_) / h_cube;

    d1_5_ = ((d1_w_5_ * ws5_).array() / h).matrix(); 
    d1w2_5_ = d1_5_(0);
    d1w3_5_ = d1_5_(1);
    d1w4_5_ = d1_5_(2);

    dws5_ = eigen_vec_c<5>{{d1_(0, 0), d1w2_5_, d1w3_5_, d1w4_5_, d1_(5, 0)}};
    // Higher order step
    // Calculate A, B
    fm_ = 1.0;
    fp_ = 1.0;
    eigen_vec_c<4> dsi_vec = dsi(d1_, d2_, d3_, ws_, gs_);
    eigen_vec_c<4> dds_vec = dds(d1_, d2_, d3_, d4w1_, ws_, gs_);
    dfpi_ = dsi_vec.sum();
    dfmi_ = std::conj(dfpi_);
    ddfp_ = dds_vec.sum() + std::pow(dsi_vec.sum(), 2);
    ddfm_ = std::conj(ddfp_);
    ap_ = (dx_ - x_ * dfmi_) / (dfpi_ - dfmi_);
    am_ = (dx_ - x_ * dfpi_) / (dfmi_ - dfpi_);
    bp_ = (ddx_ * dfmi_ - dx_ * ddfm_) / (ddfp_ * dfmi_ - ddfm_ * dfpi_);
    bm_ = (ddx_ * dfpi_ - dx_ * ddfp_) / (ddfm_ * dfpi_ - ddfp_ * dfmi_);
    dense_ap_ = ap_;
    dense_am_ = am_;
    dense_bp_ = bp_;
    dense_bm_ = bm_;
    dense_ds_i = dsi_vec;
    // Calculate step
    auto s_struct = s(h, d1_, d2_, d3_, ws_, ws5_, gs_, gs5_);
    auto dsf_vec = dsf(d1_, d2_, d3_, ws_, gs_);
    fp_ = std::exp(s_struct.val_.sum());
    fm_ = std::conj(fp_);
    dfpf_ = dsf_vec.sum() * fp_;
    dfmf_ = std::conj(dfpf_);
    // Vandermonde dense output
    Eigen::Matrix<std::complex<double>, 3, 2> result =
        Eigen::Matrix<std::complex<double>, 3, 2>::Zero();
    // Error estimate on this
    err_fp_ = s_struct.error_.cwiseAbs().sum() * fp_;
    err_fm_ = std::conj(err_fp_);
    err_dfp_ = dfpf_ / fp_ * err_fp_;
    err_dfm_ = std::conj(err_dfp_);
    result(0, 0) = ap_ * fp_ + am_ * fm_;
    result(0, 1) = bp_ * dfpf_ + bm_ * dfmf_;
    result(2, 0) = ap_ * err_fp_ + am_ * err_fm_;
    result(2, 1) = bp_ * err_dfp_ + bm_ * err_dfm_;

    // Experimental continuous representation
    // Compute some derivatives only necessary for dense output
    if constexpr (dense_output) {
      /*
      d2w2();
      d2w3();
      d2w4();
      d2w5();
      */
      //dgs_ = {d1g1_, d1g2_, d1g3_, d1g4_, d1g5_, d1g6_};
      //dgs_ = d1.col(1);

      eigen_vec_c<6> integrand6 =
          4.0 * gs_.cwiseProduct(gs_).cwiseQuotient(ws_) +
          4.0 * d1_.col(0).cwiseProduct(gs_).cwiseQuotient(ws_.cwiseProduct(ws_)) +
          d1_.col(0).cwiseProduct(d1_.col(0)).cwiseQuotient(
              ws_.cwiseProduct(ws_.cwiseProduct(ws_)));
      eigen_vec_c<6> s1_interp;
      for (int i = 0; i <= 5; i++) {
        s1_interp(i) = -1. / 2 * std::log(ws_(i));
      }
      auto ws_cwise = ws_.cwiseProduct(ws_).eval();
      eigen_vec_c<6> s2_interp =
          -1 / 4.0 *
          (d1_.col(0).cwiseQuotient(ws_cwise) + 2.0 * gs_.cwiseQuotient(ws_));
      eigen_vec_c<6> s3_interp =
          1 / 4.0 * (gs_.cwiseProduct(gs_).cwiseQuotient((ws_cwise))) +
          1 / 4.0 * (d1_.col(1).cwiseQuotient(ws_cwise)) -
          3 / 16.0 *
              (d1_.col(0).cwiseProduct(d1_.col(0)).cwiseQuotient(
                  ws_cwise.cwiseProduct(ws_cwise))) +
          1 / 8.0 * (d2_.col(0).cwiseQuotient(ws_cwise));

      // S0
      eigen_vec_c<6> s0_vdm_vec =
          h / 2.0 * std::complex<double>(0, 1) * integ_vandermonde_ * ws_;
      // S1
      eigen_vec_c<6> s1_vdm_vec = -h / 2.0 * integ_vandermonde_ * gs_;
      s1_vdm_vec.head(5) += (interp_vandermonde_ * s1_interp).tail(5);
      // S2
      eigen_vec_c<6> s2_vdm_vec =
          -h / 2.0 * 1 / 8.0 * integ_vandermonde_ * integrand6;
      s2_vdm_vec.head(5) += (interp_vandermonde_ * s2_interp).tail(5);
      // S3
      // Should the last value always be 0?
      eigen_vec_c<6> s3_vdm_vec = eigen_vec_c<6>::Zero();
      s3_vdm_vec.head(5) += (interp_vandermonde_ * s3_interp).tail(5);

      x_vdm.tail(6) = s0_vdm_vec + s1_vdm_vec +
                      std::complex<double>(0, 1) * s2_vdm_vec + s3_vdm_vec;
      x_vdm(0) = ap_;
    }

    // Lower order step for correction
    // A, B
    dsi_vec(order_) = 0.0;
    dds_vec(order_) = 0.0;
    dfpi_ = dsi_vec.sum();
    dfmi_ = std::conj(dfpi_);
    ddfp_ = dds_vec.sum() + std::pow(dsi_vec.sum(), 2);
    ddfm_ = std::conj(ddfp_);
    ap_ = (dx_ - x_ * dfmi_) / (dfpi_ - dfmi_);
    am_ = (dx_ - x_ * dfpi_) / (dfmi_ - dfpi_);
    bp_ = (ddx_ * dfmi_ - dx_ * ddfm_) / (ddfp_ * dfmi_ - ddfm_ * dfpi_);
    bm_ = (ddx_ * dfpi_ - dx_ * ddfp_) / (ddfm_ * dfpi_ - ddfp_ * dfmi_);
    // Calculate step
    s_struct.val_(order_) = 0.0;
    dsf_vec(order_) = 0.0;
    fp_ = std::exp(s_struct.val_.sum());
    fm_ = std::conj(fp_);
    dfpf_ = dsf_vec.sum() * fp_;
    dfmf_ = std::conj(dfpf_);
    result(1, 0) = result(0, 0) - ap_ * fp_ - am_ * fm_;
    result(1, 1) = result(0, 1) - bp_ * dfpf_ - bm_ * dfmf_;

    return result;
  }
  /** Computes dense output at a set of timepoints within a step.
   *
   * @param t0[in] value of independent variable (time) at the start of the step
   * @param dots[in] sequence of timepoints at which dense output is to be
   * generated
   * @param doxs[in,out] dense output for the solution \f$x(t)\f$
   * @param dodxs[in,out] dense output for the derivative of the solution
   * \f$\dot{x}\f$
   */
  void dense_step(double t0, const double h, const std::vector<double> &dots,
                  std::vector<std::complex<double>> &doxs,
                  std::vector<std::complex<double>> &dodxs) {

    // We have: ws_, gs_, ws5_, gs5_, ws7_, x, dx, ddx, h, dws_, dws5_, d2wx,
    // d3wx, etc.,

    const auto docount = dots.size();
    doxs.resize(docount);
    dodxs.resize(docount);
    // Compute some derivatives only necessary for dense output

    // Loop over dense output points
    auto dox_it = doxs.begin();
    auto dodx_it = dodxs.begin();
    auto it = dots.begin();
    for (; it != dots.end(); it++, dox_it++, dodx_it++) {
      // Transform intermediate points to be in (-1,1):
      double t_trans = 2 * (*it - t0) / h - 1;
      Eigen::Matrix<double, 6, 1> dows6 = dense_weights_6(t_trans);
      Eigen::Matrix<double, 6, 1> dodws6 = dense_weights_derivs_6(t_trans, h);

      // Dense output x
      eigen_vec_c<6> integrand6 =
          4.0 * gs_.cwiseProduct(gs_).cwiseQuotient(ws_) +
          4.0 * d1_.col(0).cwiseProduct(gs_).cwiseQuotient(ws_.cwiseProduct(ws_)) +
          d1_.col(0).cwiseProduct(d1_.col(0)).cwiseQuotient(
              ws_.cwiseProduct(ws_.cwiseProduct(ws_)));

      std::complex<double> s0 =
          std::complex<double>(0, 1) * dense_integrate(h, dows6, ws_);
      std::complex<double> s1 = dense_integrate(h, dows6, gs_);
      eigen_vec_c<6> s1_interp = (-1.0 / 2.0 * ws_.array().log()).matrix();
      s1 = dense_interpolate(dodws6, s1_interp) - s1;
      std::complex<double> s2 = dense_integrate(h, dows6, integrand6);
      eigen_vec_c<6> s2_interp = -1 / 4.0 *
                                 (d1_.col(0).cwiseQuotient(ws_.cwiseProduct(ws_)) +
                                  2.0 * gs_.cwiseQuotient(ws_));
      s2 = dense_interpolate(dodws6, s2_interp) - 1 / 8.0 * s2;

      eigen_vec_c<6> s3_interp =
          1 / 4.0 *
              (gs_.cwiseProduct(gs_).cwiseQuotient((ws_.cwiseProduct(ws_)))) +
          1 / 4.0 * (d1_.col(1).cwiseQuotient(ws_.cwiseProduct(ws_))) -
          3 / 16.0 *
              (d1_.col(0).cwiseProduct(d1_.col(0)).cwiseQuotient(
                  ws_.cwiseProduct(ws_).cwiseProduct(ws_.cwiseProduct(ws_)))) +
          1 / 8.0 *
              (d2_.col(0).cwiseQuotient(ws_.cwiseProduct(ws_).cwiseProduct(ws_)));
      std::complex<double> s3 = dense_interpolate(dodws6, s3_interp);

      std::complex<double> dense_fp = std::exp(eigen_vec_c<4>{{s0, s1, std::complex<double>(0, 1) * s2, s3}}.sum());
      std::complex<double> dense_fm = std::conj(dense_fp);
      std::complex<double> dense_x =
          dense_ap_ * dense_fp + dense_am_ * dense_fm;
      *dox_it = dense_x;

      // Same, but with Vandermonde matrix:
      double tt1 = t_trans;
      double tt2 = tt1 * t_trans;
      double tt3 = tt2 * t_trans;
      double tt4 = tt3 * t_trans;
      double tt5 = tt4 * t_trans;
      double tt6 = tt5 * t_trans;
      Eigen::Matrix<double, 6, 1> t_trans_vec = {tt1 + 1, tt2 - 1, tt3 + 1,
                                                 tt4 - 1, tt5 + 1, tt6 - 1};
      // S0
      // Vandermonde dense output
      eigen_vec_c<6> s0_vdm_vec =
          h / 2.0 * std::complex<double>(0, 1) * integ_vandermonde_ * ws_;
      // S1
      eigen_vec_c<6> s1_vdm_vec = -h / 2.0 * integ_vandermonde_ * gs_;
      s1_vdm_vec.head(5) += (interp_vandermonde_ * s1_interp).tail(5);
      // S2
      eigen_vec_c<6> s2_vdm_vec =
          -h / 2.0 * 1 / 8.0 * integ_vandermonde_ * integrand6;
      s2_vdm_vec.head(5) += (interp_vandermonde_ * s2_interp).tail(5);
      // S3
      eigen_vec_c<6> s3_vdm_vec = eigen_vec_c<6>::Zero();
      s3_vdm_vec.head(5) += (interp_vandermonde_ * s3_interp).tail(5);
      if (OSCODE_DEBUG) {
        std::complex<double> s0_vdm = t_trans_vec.dot(s0_vdm_vec);
        std::complex<double> s1_vdm = t_trans_vec.dot(s1_vdm_vec);
        std::complex<double> s2_vdm = t_trans_vec.dot(s2_vdm_vec);
        std::complex<double> s3_vdm = t_trans_vec.dot(s3_vdm_vec);
        // Delete?
        // x_vdm = dense_ap_ * (s0_vdm_vec + s1_vdm_vec + s2_vdm_vec +
        // s3_vdm_vec);

        std::cout << std::setprecision(15)
                  << "dense S0 with Vandermonde matrix: " << s0_vdm
                  << std::endl;
        std::cout << "dense S0 with classical theory: " << s0 << std::endl;
        std::cout << std::setprecision(15)
                  << "dense S1 with Vandermonde matrix: " << s1_vdm
                  << std::endl;
        std::cout << "dense S1 with classical theory: " << s1 << std::endl;
        std::cout << std::setprecision(15)
                  << "dense S2 with Vandermonde matrix: " << s2_vdm
                  << std::endl;
        std::cout << "dense S1 with classical theory: " << s2 << std::endl;
        std::cout << std::setprecision(15)
                  << "dense S3 with Vandermonde matrix: " << s3_vdm
                  << std::endl;
        std::cout << "dense S3 with classical theory: " << s3 << std::endl;

        std::cout << "Representation of S0: " << s0_vdm_vec << std::endl;
        std::cout << "Representation of S1: " << s1_vdm_vec << std::endl;
        std::cout << "Ap: " << dense_ap_ << ", Am: " << dense_am_ << std::endl;
      }

      // Dense output dx
      std::complex<double> ds0 =
          std::complex<double>(0, 1) * dense_interpolate(dodws6, ws_);
      std::complex<double> ds1 = dense_interpolate(dodws6, gs_);
      eigen_vec_c<6> ds1_interp = -1. / 2 * d1_.col(0).cwiseQuotient(ws_);
      ds1 = dense_interpolate(dodws6, ds1_interp) - ds1;
      eigen_vec_c<6> ds2_interp =
          -1. / 2 * gs_.cwiseProduct(gs_.cwiseQuotient(ws_)) -
          1. / 2 * d1_.col(1).cwiseQuotient(ws_) +
          3. / 8 *
              (d1_.col(0).cwiseProduct(d1_.col(0)))
                  .cwiseQuotient((ws_.cwiseProduct(ws_)).cwiseProduct(ws_)) -
          1. / 4 * d2_.col(0).cwiseQuotient(ws_.cwiseProduct(ws_));
      std::complex<double> ds2 = dense_interpolate(dodws6, ds2_interp);
      eigen_vec_c<6> ds3_interp =
          1. / 8.0 *
          (d2_.col(1).cwiseQuotient(ws_.cwiseProduct(ws_.cwiseProduct(ws_))) +
           2 * d2_.col(1).cwiseQuotient(ws_.cwiseProduct(ws_)) -
           6 * (d1_.col(0).cwiseProduct(d2_.col(0)))
                   .cwiseQuotient(ws_.cwiseProduct(
                       ws_.cwiseProduct(ws_.cwiseProduct(ws_)))) +
           6 * (d1_.col(0).cwiseProduct(d1_.col(0).cwiseProduct(d1_.col(0))))
                   .cwiseQuotient(ws_.cwiseProduct(ws_.cwiseProduct(
                       ws_.cwiseProduct(ws_.cwiseProduct(ws_))))) -
           4 * (gs_.cwiseProduct(gs_) + d1_.col(1))
                   .cwiseProduct(d1_.col(0))
                   .cwiseQuotient(ws_.cwiseProduct(ws_.cwiseProduct(ws_))) +
           4 * d1_.col(1).cwiseProduct(gs_).cwiseQuotient(
                   ws_.cwiseProduct(ws_.cwiseProduct(ws_.cwiseProduct(ws_)))));
      std::complex<double> ds3 = dense_interpolate(dodws6, ds3_interp);
      dense_ds_ =
          eigen_vec_c<4>{{ds0, ds1, std::complex<double>(0, 1) * ds2, ds3}} +
          dense_ds_i;
      std::complex<double> dense_dfpf = dense_ds_.sum() * dense_fp;
      std::complex<double> dense_dfmf = std::conj(dense_dfpf);
      std::complex<double> dense_dx =
          dense_bp_ * dense_dfpf + dense_bm_ * dense_dfmf;
      *dodx_it = dense_dx;
    }

    return;
  }
  // Compute integration weights at a given point
  inline Eigen::Matrix<double, 6, 1> dense_weights_6(double t) const noexcept {

    double a = std::sqrt(147 + 42 * std::sqrt(7.));
    double b = std::sqrt(147 - 42 * std::sqrt(7.));
    double c = (-2. / 35.0 * t * t * t + 4. / 35.0 * t * t - 2. / 45.0 * t -
                8. / 315.0) *
               std::sqrt(7.);
    double w1 = 31.0 / 480.0 - 7. / 32.0 * t * t * t * t * t * t +
                21. / 80.0 * t * t * t * t * t + 7. / 32.0 * t * t * t * t -
                7. / 24.0 * t * t * t - 1. / 32.0 * t * t + 1. / 16.0 * t;
    double w2 = -2205.0 *
                ((c - 4. / 63.0 * t + 8. / 63.0) * a +
                 (t - 1.0) * (t - 1.0) * (t * t * std::sqrt(7.) + 1.0)) *
                (t + 1.0) * (t + 1.0) / (a * (-1120.0 + 160.0 * std::sqrt(7.)));
    double w3 = -2205.0 *
                ((c + 4. / 63.0 * t - 8. / 63.0) * b +
                 (t - 1.0) * (t - 1.0) * (t * t * std::sqrt(7.) - 1.0)) *
                (t + 1.0) * (t + 1.0) / (b * (1120.0 + 160.0 * std::sqrt(7.)));
    double w4 = 2205.0 *
                ((-c - 4. / 63.0 * t + 8. / 63.0) * b +
                 (t - 1.0) * (t - 1.0) * (t * t * std::sqrt(7.) - 1.0)) *
                (t + 1.0) * (t + 1.0) / (b * (1120.0 + 160.0 * std::sqrt(7.)));
    double w5 = 2205.0 *
                ((-c + 4. / 63.0 * t - 8. / 63.0) * a +
                 (t - 1.0) * (t - 1.0) * (t * t * std::sqrt(7.) + 1.0)) *
                (t + 1.0) * (t + 1.0) / (a * (-1120.0 + 160.0 * std::sqrt(7.)));
    double w6 = 1. / 480.0 + 7. / 32.0 * t * t * t * t * t * t +
                21. / 80.0 * t * t * t * t * t - 7. / 32.0 * t * t * t * t -
                7. / 24.0 * t * t * t + 1. / 32.0 * t * t + 1. / 16.0 * t;
    return {w1, w2, w3, w4, w5, w6};
  }

  // Compute weights of the interpolating polynomial
  inline Eigen::Matrix<double, 6, 1>
  dense_weights_derivs_6(double t, const double h) const noexcept {

    double a = std::sqrt(147.0 + 42.0 * std::sqrt(7.));
    double b = std::sqrt(147.0 - 42.0 * std::sqrt(7.));
    double c = (27783.0 * t * t * t * t * t * t - 46305.0 * t * t * t * t +
                19845.0 * t * t - 1323.0) *
               sqrt(7.) / 16.;
    double w1 = -(21.0 * t * t * t * t - 14.0 * t * t + 1) * (t - 1) / 16.;
    double w2 = -c / (a * (-7.0 + sqrt(7.)) * (21.0 * t + a));
    double w3 = -c / (b * (7.0 + sqrt(7.)) * (21.0 * t + b));
    double w4 = c / (b * (7.0 + sqrt(7.)) * (21.0 * t - b));
    double w5 = c / (a * (-7.0 + sqrt(7.)) * (21.0 * t - a));
    double w6 = (21.0 * t * t * t * t - 14.0 * t * t + 1.0) * (t + 1.0) / 16.;
    return {w1, w2, w3, w4, w5, w6};
  }
  inline std::complex<double> dense_integrate(const double h,
      const Eigen::Matrix<double, 6, 1> &denseweights6,
      const Eigen::Matrix<std::complex<double>, 6, 1> &integrand6) const {

    return h / 2.0 * denseweights6.dot(integrand6);
  }
  inline std::complex<double>
  dense_interpolate(const Eigen::Matrix<double, 6, 1> &denseweights6,
                    const Eigen::Matrix<std::complex<double>, 6, 1> &integrand6)
      const noexcept {

    Eigen::Matrix<double, 6, 1> mod_weights = denseweights6;
    mod_weights(0) -= 1.0;
    return mod_weights.dot(integrand6);
  }

};



//////////////////////////////////

class WKBSolver1 : public WKBSolver {
private:
  virtual eigen_vec_c<4> dds(const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const std::complex<double>& d4_w1,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs) final {
    return eigen_vec_c<4>{{std::complex<double>(0, 1) * d1(1 - 1, 0),
                        1.0 / std::pow(ws(0), 2) * std::pow(d1(1 - 1, 0), 2) / 2.0 -
                            1.0 / ws(0) * d2(1 - 1, 0) / 2.0 - d1(1 - 1, 1),
                        0.0, 0.0}};
  }
  virtual eigen_vec_c<4> dsi(const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs) final {
    return eigen_vec_c<4>{{std::complex<double>(0.0, 1.0) * ws(0),
                           -0.5 * d1(1 - 1, 0) / ws(0) - gs(0), 0.0, 0.0}};
  }
  virtual eigen_vec_c<4> dsf(const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs) final {
    return eigen_vec_c<4>{{std::complex<double>(0.0, 1.0) * ws(5),
                           -0.5 * d1(6 - 1, 0) / ws(5) - gs(5), 0.0, 0.0}};
  }
  virtual series s(const double h, const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 5, 1>& ws5,
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs,
    const Eigen::Matrix<std::complex<double>, 5, 1>& gs5) final {
    eigen_vec_c<2> s0 = std::complex<double>(0, 1) * integrate(h, ws, ws5);
    eigen_vec_c<2> s1 = integrate(h, gs, gs5_);
    s1(0) = std::log(std::sqrt(ws(0) / ws(5))) - s1(0);
    return series{eigen_vec_c<4>{{s0(0), s1(0), 0.0, 0.0}}, eigen_vec_c<4>{{s0(1), s1(1), 0.0, 0.0}}};
  }

public:
  WKBSolver1(){};
  WKBSolver1(de_system &de_sys, int order) : WKBSolver(de_sys, order){};
};

//////////////////////////////////

class WKBSolver2 : public WKBSolver {
private:
  virtual eigen_vec_c<4> dds(const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const std::complex<double>& d4_w1,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs) final {
    return eigen_vec_c<4>{
        {std::complex<double>(0, 1) * d1(1 - 1, 0),
         1.0 / std::pow(ws(0), 2) * std::pow(d1(1 - 1, 0), 2) / 2.0 -
             1.0 / ws(0) * d2(1 - 1, 0) / 2.0 - d1(1 - 1, 1),
         -std::complex<double>(0, 1 / 8) *
             (8.0 * d1(1 - 1, 1) * gs(0) * std::pow(ws(0), 3) -
              4.0 * d1(1 - 1, 0) * std::pow(gs(0), 2) * std::pow(ws(0), 2) +
              4.0 * d2(1 - 1, 1) * std::pow(ws(0), 3) -
              4.0 * d1(1 - 1, 0) * d1(1 - 1, 1) * std::pow(ws(0), 2) +
              2.0 * d2(1 - 1, 0) * std::pow(ws(0), 2) -
              10.0 * d1(1 - 1, 0) * d2(1 - 1, 0) * ws(0) + 9.0 * std::pow(d1(1 - 1, 0), 3)) /
             std::pow(ws(0), 4),
         0.0}};
  }
  virtual eigen_vec_c<4> dsi(const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs) final {
    return eigen_vec_c<4>{{std::complex<double>(0, 1) * ws(0),
                        -1.0 / ws(0) * d1(1 - 1, 0) / 2.0 - gs(0),
                        std::complex<double>(0, 1 / 8) *
                            (-4.0 * std::pow(gs(0), 2) * std::pow(ws(0), 2) -
                             4.0 * d1(1 - 1, 1) * std::pow(ws(0), 2) -
                             2.0 * d2(1 - 1, 0) * ws(0) + 3.0 * std::pow(d1(1 - 1, 0), 2)) /
                            std::pow(ws(0), 3),
                        0.0}};
  }
  virtual eigen_vec_c<4> dsf(const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs) final {
    return eigen_vec_c<4>{{std::complex<double>(0, 1) * ws(5),
                        -1.0 / ws(5) * d1(6 - 1, 0) / 2.0 - gs(5),
                        std::complex<double>(0, 1 / 8) *
                            (-4.0 * std::pow(gs(5), 2) * std::pow(ws(5), 2) -
                             4.0 * d1(6 - 1, 1) * std::pow(ws(5), 2) -
                             2.0 * d2(6 - 1, 0) * ws(5) + 3.0 * std::pow(d1(6 - 1, 0), 2)) /
                            std::pow(ws(5), 3),
                        0.0}};
  }
  virtual series s(const double h, const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 5, 1>& ws5,
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs,
    const Eigen::Matrix<std::complex<double>, 5, 1>& gs5) final {
    eigen_vec_c<6> integrand6 =
        4 * gs.cwiseProduct(gs).cwiseQuotient(ws) +
        4 * d1_.col(0).cwiseProduct(gs).cwiseQuotient(ws.cwiseProduct(ws)) +
        d1_.col(0).cwiseProduct(d1_.col(0)).cwiseQuotient(
            ws.cwiseProduct(ws.cwiseProduct(ws)));
    eigen_vec_c<5> integrand5 =
        4 * gs5_.cwiseProduct(gs5_).cwiseQuotient(ws5) +
        4 * dws5_.cwiseProduct(gs5_).cwiseQuotient(ws5.cwiseProduct(ws5)) +
        dws5_.cwiseProduct(dws5_).cwiseQuotient(
            ws5.cwiseProduct(ws5.cwiseProduct(ws5)));
    eigen_vec_c<2> s0 = std::complex<double>(0, 1) * integrate(h, ws, ws5);
    eigen_vec_c<2> s1 = integrate(h, gs, gs5_);
    s1(0) = std::log(std::sqrt(ws(0) / ws(5))) - s1(0);
    eigen_vec_c<2> s2 = integrate(h, integrand6, integrand5);
    s2(0) = -1 / 4.0 *
                (d1_.col(0)(5) / std::pow(ws(5), 2) + 2.0 * gs(5) / ws(5) -
                 d1_.col(0)(0) / std::pow(ws(0), 2) - 2.0 * gs(0) / ws(0)) -
            1 / 8.0 * s2(0);
    return {eigen_vec_c<4>{{s0(0), s1(0), std::complex<double>(0, 1) * s2(0), 0.0}}, 
     eigen_vec_c<4>{{s0(1), s1(1), std::complex<double>(0, -1 / 8) * s2(1), 0.0}}};
  }

public:
  WKBSolver2(){};
  WKBSolver2(de_system &de_sys, int order) : WKBSolver(de_sys, order) {}
};

//////////////////////////////////

class WKBSolver3 : public WKBSolver {
private:
  virtual eigen_vec_c<4> dds(const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const std::complex<double>& d4_w1,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs) final {
    return eigen_vec_c<4>{
        {std::complex<double>(0, 1) * d1(1 - 1, 0),
         1.0 / (ws(0) * ws(0)) * (d1(1 - 1, 0) * d1(1 - 1, 0)) / 2.0 -
             1.0 / ws(0) * d2(1 - 1, 0) / 2.0 - d1(1 - 1, 1),
         -std::complex<double>(0, 1.0 / 8.0) *
             (8.0 * d1(1 - 1, 1) * gs(0) * (ws(0) * ws(0) * ws(0)) -
              4.0 * d1(1 - 1, 0) * (gs(0) * gs(0)) * ws(0) * ws(0) +
              4.0 * d2(1 - 1, 1) * (ws(0) * ws(0) * ws(0)) -
              4.0 * d1(1 - 1, 0) * d1(1 - 1, 1) * ws(0) * ws(0) +
              2.0 * d2(1 - 1, 0) * ws(0) * ws(0) - 10.0 * d1(1 - 1, 0) * d2(1 - 1, 0) * ws(0) +
              9.0 * (d1(1 - 1, 0) * d1(1 - 1, 0) * d1(1 - 1, 0))) /
             (ws(0) * ws(0) * ws(0) * ws(0)),
         (d4w1_ * (ws(0) * ws(0) * ws(0)) +
          2.0 * d3g1_ * (ws(0) * ws(0) * ws(0) * ws(0)) -
          9.0 * d1(1 - 1, 0) * d2(1 - 1, 0) * ws(0) * ws(0) -
          6.0 * (d2(1 - 1, 0) * d2(1 - 1, 0)) * ws(0) * ws(0) +
          (42.0 * ws(0) * (d1(1 - 1, 0) * d1(1 - 1, 0)) -
           4.0 * (ws(0) * ws(0) * ws(0)) * ((gs(0) * gs(0)) + d1(1 - 1, 1))) *
              d2(1 - 1, 0) +
          (4.0 * gs(0) * (ws(0) * ws(0) * ws(0) * ws(0)) -
           8.0 * (ws(0) * ws(0) * ws(0)) * d1(1 - 1, 0)) *
              d2(1 - 1, 1) -
          30.0 * (d1(1 - 1, 0) * d1(1 - 1, 0) * d1(1 - 1, 0) * d1(1 - 1, 0)) +
          12.0 * ws(0) * ws(0) * ((gs(0) * gs(0)) + d1(1 - 1, 1)) *
              (d1(1 - 1, 0) * d1(1 - 1, 0)) -
          16.0 * d1(1 - 1, 0) * d1(1 - 1, 1) * gs(0) * (ws(0) * ws(0) * ws(0)) +
          4.0 * (d1(1 - 1, 1) * d1(1 - 1, 1)) * (ws(0) * ws(0) * ws(0) * ws(0))) /
             (ws(0) * ws(0) * ws(0) * ws(0) * ws(0) * ws(0)) / 8.0}};
  }
  virtual eigen_vec_c<4> dsi(const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs) final {
    return eigen_vec_c<4>{
        {std::complex<double>(0, 1) * ws(0),
         -1.0 / ws(0) * d1(1 - 1, 0) / 2.0 - gs(0),
         std::complex<double>(0, 1.0 / 8.0) *
             (-4.0 * (gs(0) * gs(0)) * ws(0) * ws(0) -
              4.0 * d1(1 - 1, 1) * ws(0) * ws(0) - 2.0 * d2(1 - 1, 0) * ws(0) +
              3.0 * (d1(1 - 1, 0) * d1(1 - 1, 0))) /
             (ws(0) * ws(0) * ws(0)),
         (d2(1 - 1, 0) * ws(0) * ws(0) + 2.0 * d2(1 - 1, 1) * (ws(0) * ws(0) * ws(0)) -
          6.0 * d1(1 - 1, 0) * d2(1 - 1, 0) * ws(0) + 6.0 * (d1(1 - 1, 0) * d1(1 - 1, 0) * d1(1 - 1, 0)) -
          4.0 * ((gs(0) * gs(0)) + d1(1 - 1, 1)) * ws(0) * ws(0) * d1(1 - 1, 0) +
          4.0 * d1(1 - 1, 1) * gs(0) * (ws(0) * ws(0) * ws(0))) /
             (ws(0) * ws(0) * ws(0) * ws(0) * ws(0)) / 8.0}};
  }
  virtual eigen_vec_c<4> dsf(const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs) final {
    return eigen_vec_c<4>{
        {std::complex<double>(0, 1) * ws(5),
         -1.0 / ws(5) * d1(6 - 1, 0) / 2.0 - gs(5),
         std::complex<double>(0, 1.0 / 8.0) *
             (-4.0 * (gs(5) * gs(5)) * (ws(5) * ws(5)) -
              4.0 * d1(6 - 1, 1) * (ws(5) * ws(5)) - 2.0 * d2(6 - 1, 0) * ws(5) +
              3.0 * (d1(6 - 1, 0) * d1(6 - 1, 0))) /
             (ws(5) * ws(5) * ws(5)),
         (d2(6 - 1, 0) * (ws(5) * ws(5)) + 2.0 * d2(6 - 1, 1) * (ws(5) * ws(5) * ws(5)) -
          6.0 * d1(6 - 1, 0) * d2(6 - 1, 0) * ws(5) + 6.0 * (d1(6 - 1, 0) * d1(6 - 1, 0) * d1(6 - 1, 0)) -
          4.0 * ((gs(5) * gs(5)) + d1(6 - 1, 1)) * (ws(5) * ws(5)) * d1(6 - 1, 0) +
          4.0 * d1(6 - 1, 1) * gs(5) * (ws(5) * ws(5) * ws(5))) /
             (ws(5) * ws(5) * ws(5) * ws(5) * ws(5)) / 8.0}};
  }
  virtual series s(double h, const Eigen::Matrix<std::complex<double>, 6, 2>& d1, 
    const Eigen::Matrix<std::complex<double>, 6, 2>& d2, 
    const Eigen::Matrix<std::complex<double>, 6, 1>& d3,
    const Eigen::Matrix<std::complex<double>, 6, 1>& ws, 
    const Eigen::Matrix<std::complex<double>, 5, 1>& ws5,
    const Eigen::Matrix<std::complex<double>, 6, 1>& gs,
    const Eigen::Matrix<std::complex<double>, 5, 1>& gs5) final {
    auto ws_cwise = ws.cwiseProduct(ws).eval();
    eigen_vec_c<6> integrand6 =
        4.0 * gs.cwiseProduct(gs).cwiseQuotient(ws) +
        4.0 * d1.col(0).cwiseProduct(gs).cwiseQuotient(ws_cwise) +
        d1.col(0).cwiseProduct(d1.col(0)).cwiseQuotient(ws.cwiseProduct(ws_cwise));
    auto ws5_cwise = ws5.cwiseProduct(ws5);
    eigen_vec_c<5> integrand5 =
        4.0 * gs5.cwiseProduct(gs5).cwiseQuotient(ws5) +
        4.0 * dws5_.cwiseProduct(gs5).cwiseQuotient(ws5_cwise) +
        dws5_.cwiseProduct(dws5_).cwiseQuotient(ws5.cwiseProduct(ws5_cwise));
    eigen_vec_c<2> s0 = std::complex<double>(0, 1) * integrate(h, ws, ws5);
    eigen_vec_c<2> s1 = integrate(h, gs, gs5);
    s1(0) = std::log(std::sqrt(ws(0) / ws(5))) - s1(0);
    eigen_vec_c<2> s2 = integrate(h, integrand6, integrand5);
    s2(0) = -1 / 4.0 *
                (d1.col(0)(5) / (ws(5) * ws(5)) + 2.0 * gs(5) / ws(5) -
                 d1.col(0)(0) / (ws(0) * ws(0)) - 2.0 * gs(0) / ws(0)) -
            1 / 8.0 * s2(0);
    std::complex<double> s3 =
        (1 / 4.0 *
             (gs(5) * gs(5) / (ws(5) * ws(5)) -
              gs(0) * gs(0) / (ws(0) * ws(0))) +
         1 / 4.0 * (d1(6 - 1, 1) / (ws(5) * ws(5)) - d1(1 - 1, 1) / (ws(0) * ws(0))) -
         3 / 16.0 *
             (d1.col(0)(5) * d1.col(0)(5) / (ws(5) * ws(5) * ws(5) * ws(5)) -
              d1.col(0)(0) * d1.col(0)(0) / (ws(0) * ws(0) * ws(0) * ws(0))) +
         1 / 8.0 *
             (d2(6 - 1, 0) / (ws(5) * ws(5) * ws(5)) -
              d2(1 - 1, 0) / (ws(0) * ws(0) * ws(0))));
    return {eigen_vec_c<4>{{s0(0), s1(0), std::complex<double>(0, 1) * s2(0), s3}}, 
     eigen_vec_c<4>{{s0(1), s1(1), std::complex<double>(0, -1.0 / 8.0) * s2(1), 0.0}}};
  }

public:
  WKBSolver3() {}
  WKBSolver3(de_system &de_sys, int order) : WKBSolver(de_sys, order) {}
};

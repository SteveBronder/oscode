#pragma once
#include <oscode/interpolator.hpp>
#include <oscode/rksolver.hpp>
#include <oscode/system.hpp>
#include <oscode/wkbsolver.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <vector>

/** A class to store all information related to a numerical solution run.  */
template <typename Sys> class Solution {
private:
  double t, tf, rtol, atol, h0;
  std::complex<double> x, dx;
  int order;
  const char *fo;
  std::unique_ptr<WKBSolver> wkbsolver;
  /** A \a de_system object to carry information about the ODE. */
  Sys *de_sys_;
  /** These define the event at which integration finishes (currently: when tf
   * is reached. */
  double fend, fnext;
  /** a boolean encoding the direction of integration: 1/True for forward. */
  bool sign;
  bool dense_output_{false};
  auto set_wkb_solver_order(int order, Sys *&desys) {
    switch (order) {
    case 1:
      return std::unique_ptr<WKBSolver>(new WKBSolver1(order));
    case 2:
      return std::unique_ptr<WKBSolver>(new WKBSolver2(order));
    case 3:
      return std::unique_ptr<WKBSolver>(new WKBSolver3(order));
    };
    return std::unique_ptr<WKBSolver>(new WKBSolver3(order));
  }
public:

  /** Object to call RK steps */
  RKSolver<Sys> rksolver;

  /** Successful, total attempted, and successful WKB steps the solver took,
   * respectively  */
  int ssteps, totsteps, wkbsteps;
  /** Lists to contain the solution and its derivative evaluated at internal
   * points taken by the solver (i.e. not dense output) after a run */
  std::vector<std::complex<double>> sol, dsol;
  /** List to contain the timepoints at which the solution and derivative are
   * internally evaluated by the solver */
  std::vector<double> times;
  /** List to contain the "type" of each step (RK/WKB) taken internally by the
   * solver after a run */
  std::vector<bool> wkbs;
  /** Lists to contain the timepoints at which dense output was evaluated. This
   * list will always be sorted in ascending order (with possible duplicates),
   * regardless of the order the timepoints were specified upon input. */
  std::vector<double> dotimes;
  /** Lists to contain the dense output of the solution and its derivative */
  std::vector<std::complex<double>> dosol, dodsol;
  /** Iterator to iterate over the dense output timepoints, for when these
   * need to be written out to file */
  // std::vector<double>::iterator dotit;
  //  Experimental: list to contain continuous representation of the solution
  std::vector<Eigen::Matrix<std::complex<double>, 7, 1>> sol_vdm;

  /** Constructor for when dense output was not requested. Sets up solution of
   * the ODE.
   *
   * @param[in] de_sys de_system object carrying information about the ODE being
   * solved
   * @param[in] x0, dx0 initial conditions for the ODE, \f$ x(t) \f$, \f$
   * \frac{dx}{dt} \f$ evaluated at the start of the integration range
   * @param[in] t_i start of integration range
   * @param[in] t_f end of integration range
   * @param[in] o order of WKB approximation to be used
   * @param[in] r_tol (local) relative tolerance
   * @param[in] a_tol (local) absolute tolerance
   * @param[in] h_0 initial stepsize to use
   * @param[in] full_output file name to write results to
   *
   */
  Solution(Sys &de_sys, std::complex<double> x0, std::complex<double> dx0,
           double t_i, double t_f, int o = 3, double r_tol = 1e-4,
           double a_tol = 0, double h_0 = 1, const char *full_output = ""): 
           t(t_i), tf(t_f), rtol(r_tol), atol(a_tol), 
           h0(h_0), x(x0), dx(dx0), order(o), fo(full_output), wkbsolver(set_wkb_solver_order(order, de_sys_)),
           de_sys_(&de_sys),  fend(0), fnext(0), sign(false),
           dense_output_(false),
           rksolver(*de_sys_) ,
           ssteps(0), totsteps(0), wkbsteps(0), 
           sol(0), dsol(0), times(0), wkbs(0), dotimes(0), dosol(0), dodsol(0), sol_vdm(0)

           {


    // Determine direction of integration, fend>0 and integration ends when
    // it crosses zero
    if ((t >= tf) && h0 < 0) {
      // backwards
      fend = t - tf;
      fnext = fend;
      de_sys_->w_func_.sign_ = !de_sys_->is_interpolated;
      de_sys_->g_func_.sign_ = !de_sys_->is_interpolated;
    } else if ((t <= tf) && h0 > 0) {
      // forward
      fend = tf - t;
      fnext = fend;
      de_sys_->w_func_.sign_ = de_sys_->is_interpolated;
      de_sys_->g_func_.sign_ = de_sys_->is_interpolated;
      sign = !de_sys_->is_interpolated;
    } else {
      throw std::logic_error(
          "Direction of integration in conflict with direction of initial "
          "step, "
          "terminating. Please check your values for ti, tf, and h.");
      return;
    }

    // No dense output desired if this constructor was called, so only output
    // answer at t_i and t_f
    dotimes.push_back(t_i);
    dotimes.push_back(t_f);
    dosol.push_back(x0);
    dodsol.push_back(dx0);
 
  }

  /** Constructor for when dense output was requested. Sets up solution of the
   * ODE.
   *
   * @param[in] de_sys de_system object carrying information about the ODE being
   * solved
   * @param[in] x0, dx0 initial conditions for the ODE, \f$ x(t) \f$, \f$
   * \frac{dx}{dt} \f$ evaluated at the start of the integration range
   * @param[in] t_i start of integration range
   * @param[in] t_f end of integration range
   * @param[in] do_times timepoints at which dense output is to be produced.
   * Doesn't need to be sorted, and duplicated are allowed.
   * @param[in] o order of WKB approximation to be used
   * @param[in] r_tol (local) relative tolerance
   * @param[in] a_tol (local) absolute tolerance
   * @param[in] h_0 initial stepsize to use
   * @param[in] full_output file name to write results to
   *
   */
  template <typename X>
  Solution(Sys &de_sys, std::complex<double> x0, std::complex<double> dx0,
           double t_i, double t_f, X&& do_times, int o = 3,
           double r_tol = 1e-4, double a_tol = 0, double h_0 = 1,
           const char *full_output = "") :
                      t(t_i), tf(t_f), rtol(r_tol), atol(a_tol), 
           h0(h_0), x(x0), dx(dx0), order(o), fo(full_output), wkbsolver(set_wkb_solver_order(order, de_sys_)),
           de_sys_(&de_sys),  fend(0), fnext(0), sign(false),
           dense_output_(true), rksolver(*de_sys_),
            ssteps(0), totsteps(0), wkbsteps(0), 
           sol(0), dsol(0), times(0), wkbs(0), dotimes(std::forward<X>(do_times)), dosol(0), dodsol(0), sol_vdm(0)
            {


    // Determine direction of integration, fend>0 and integration ends when
    // it crosses zero
    if ((t >= tf) && h0 < 0) {
      // backwards
      fend = t - tf;
      fnext = fend;
      de_sys_->w_func_.sign_ = !de_sys_->is_interpolated;
      de_sys_->g_func_.sign_ = !de_sys_->is_interpolated;
    } else if ((t <= tf) && h0 > 0) {
      // forward
      fend = tf - t;
      fnext = fend;
      de_sys_->w_func_.sign_ = de_sys_->is_interpolated;
      de_sys_->g_func_.sign_ = de_sys_->is_interpolated;
      sign = !de_sys_->is_interpolated;
    } else {
      throw std::logic_error(
          "Direction of integration in conflict with direction of initial "
          "step, "
          "terminating. Please check your values for ti, tf, and h.");
      return;
    }

    // Dense output preprocessing: sort and reverse if necessary
    const std::size_t dosize = do_times.size();
    dosol.resize(dosize);
    dodsol.resize(dosize);

     // Sort to ensure ascending order
    std::sort(dotimes.begin(), dotimes.end());

    // Reverse if necessary
    if ((de_sys_->is_interpolated && !de_sys_->w_func_.sign_) ||
        (!de_sys_->is_interpolated && !sign)) {
      std::reverse(dotimes.begin(), dotimes.end());
    }
  }
  /** \brief Function to solve the ODE \f$ \ddot{x} + 2\gamma(t)\dot{x} +
   * \omega^2(t)x = 0 \f$ for \f$ x(t), \frac{dx}{dt} \f$.
   *
   * While solving the ODE, this function will populate the \a Solution object
   * with the following results:
   *
   */
  void solve() {

    // Settings for MS
    static constexpr int nrk = 5;
    static constexpr int nwkb1 = 2;
    static constexpr int nwkb2 = 4;
 
    Eigen::Matrix<double, 1, 2> errmeasure_rk;
    Eigen::Matrix<double, 1, 4> errmeasure_wkb;
    double tnext, hnext, h, hrk, hwkb;
    double wkbdelta, rkdelta;
    std::complex<double> xnext, dxnext;
    bool wkb = false;
    Eigen::Index maxindex_wkb{0}, maxindex_rk;
    h = h0;
    tnext = t + h;
    // Initialise stats
    sol.push_back(x);
    dsol.push_back(dx);
    times.push_back(t);
    wkbs.push_back(false);
    ssteps = 0;
    totsteps = 0;
    wkbsteps = 0;
    // Dense output
    std::vector<double> inner_dotimes;
    std::vector<std::complex<double>> inner_dosols, inner_dodsols;
    Eigen::Matrix<std::complex<double>, 1, 2> y_dense_rk;
    std::complex<double> x_dense_rk, dx_dense_rk;
    // Experimental continuous solution, vandermonde representation
    Eigen::Matrix<std::complex<double>, 7, 1> xvdm;
    std::size_t do_solve_count = 0;
    std::size_t do_dodsol_count = 0;
    bool warn_once = true;

    while (fend > 0) {
      // Check if we are reaching the end of integration
      if (fnext < 0) {
        h = tf - t;
        tnext = tf;
      }
      // Keep updating stepsize until step is accepted
      while (true) {
        // RK step
        auto rkstep = rksolver.step(x, dx, t, h);
        auto rkx = rkstep.col(0);
        auto rkerr = rkstep.col(1);
        // WKB step
        auto wkbstep = wkbsolver->step(dense_output_, x, dx, t, h, rksolver.ws_gs_ ,
                                  rksolver.ws5_, rksolver.gs5_);
        auto wkbx = wkbstep.col(0);
        auto wkberr = wkbstep.col(2);
        auto truncerr = wkbstep.col(1);
        // Safety feature for when all wkb steps are 0 (truncer=0), but not
        // necessarily in good WKB regime:
        truncerr(0) = std::max(1e-10, abs(truncerr(0)));
        truncerr(1) = std::max(1e-10, abs(truncerr(1)));
        // dominant error calculation
        // Error scale measures
        errmeasure_rk << std::abs(rkerr(0)) / (std::abs(rkx(0)) * rtol + atol),
            std::abs(rkerr(1)) / (std::abs(rkx(1)) * rtol + atol);
        errmeasure_wkb << std::abs(truncerr(0)) /
                              (std::abs(wkbx(0)) * rtol + atol),
            std::abs(truncerr(1)) / (std::abs(wkbx(1)) * rtol + atol),
            std::abs(wkberr(0)) / (std::abs(wkbx(0)) * rtol + atol),
            std::abs(wkberr(1)) / (std::abs(wkbx(1)) * rtol + atol);
        rkdelta = std::max(1e-10, errmeasure_rk.maxCoeff(&maxindex_rk));
        if (std::isnan(errmeasure_wkb.maxCoeff()) == false &&
            std::isinf(std::real(wkbx(0))) == false &&
            std::isinf(std::imag(wkbx(0))) == false &&
            std::isinf(std::real(wkbx(1))) == false &&
            std::isinf(std::imag(wkbx(1))) == false &&
            std::isnan(std::real(wkbx(0))) == false &&
            std::isnan(std::imag(wkbx(0))) == false &&
            std::isnan(std::real(wkbx(1))) == false &&
            std::isnan(std::imag(wkbx(1))) == false) {
          wkbdelta = std::max(1e-10, errmeasure_wkb.maxCoeff(&maxindex_wkb));
        } else {
          wkbdelta = std::numeric_limits<double>::infinity();
        }

        // predict next stepsize
        hrk = h * std::pow((1.0 / rkdelta), 1.0 / nrk);
        hwkb = h * std::pow(1.0 / wkbdelta, 1.0 / (maxindex_wkb <= 1 ? nwkb1 : nwkb2)); 


        // choose step with larger predicted stepsize
        wkb = std::abs(hwkb) >= std::abs(hrk);
        if (wkb) {
          xnext = wkbx(0);
          dxnext = wkbx(1);
          // if wkb step chosen, ignore truncation error in
          // stepsize-increase
          wkbdelta = std::max(1e-10, errmeasure_wkb.tail(2).maxCoeff());
          hnext = h * std::pow(1.0 / wkbdelta, 1.0 / nwkb2);
        } else {
          xnext = rkx(0);
          dxnext = rkx(1);
          hnext = hrk;
        }
        // check if chosen step was successful
        if (std::abs(hnext) >= std::abs(h)) {
          if (dense_output_ && do_solve_count < dotimes.size()) {
            for (auto dot_it = dotimes.begin() + do_solve_count;
             dot_it != dotimes.end() &&
             ((*dot_it - t >= 0 && tnext - *dot_it >= 0) ||
              (*dot_it - t <= 0 && tnext - *dot_it <= 0));
              ++dot_it) {
              inner_dotimes.push_back(*dot_it);
              do_solve_count++;
            }
            if (inner_dotimes.size() > 0) {
              inner_dosols.resize(inner_dotimes.size());
              inner_dodsols.resize(inner_dotimes.size());
              if (wkb) {
                wkbsolver->dense_step(t, h, inner_dotimes, rksolver.ws_gs_, rksolver.ws5_, rksolver.gs5_,inner_dosols, 
                                      inner_dodsols);
              } else {
                for (auto it = inner_dotimes.begin(); it != inner_dotimes.end();
                     it++)
                  rksolver.dense_step(t, h, x, dx, inner_dotimes, inner_dosols,
                                      inner_dodsols);
              }
            }
          }
          auto it_dosol = dosol.begin();
          std::advance(it_dosol, do_dodsol_count);
          auto it_dodsol = dodsol.begin();
          std::advance(it_dodsol, do_dodsol_count);
          for (auto inner_it = inner_dosols.begin(),
                    inner_dit = inner_dodsols.begin();
               inner_it != inner_dosols.end() && it_dosol != dosol.end() &&
               inner_dit != inner_dodsols.end() && it_dodsol != dodsol.end();
               it_dodsol++, it_dosol++, inner_it++, inner_dit++,
                    do_dodsol_count++) {
            *it_dosol = std::move(*inner_it);
            *it_dodsol = std::move(*inner_dit);
          }
          
          inner_dotimes.clear();
          inner_dosols.clear();
          inner_dodsols.clear();

          // record type of step
          wkbsteps += wkb;
          wkbs.push_back(wkb);
          xvdm = wkb ? rksolver.x_vdm_ : wkbsolver->x_vdm;
            
          sol.push_back(xnext);
          dsol.push_back(dxnext);
          sol_vdm.push_back(xvdm);
          times.push_back(tnext);
          tnext += hnext;
          x = xnext;
          dx = dxnext;
          t += h;
          h = hnext;
          fend = h > 0 ? tf - t : t - tf;
          fnext = h > 0 ? tf - tnext : tnext - tf;
          ssteps += 1;
          // Update interpolation bounds
          de_sys_->w_func_.update_interp_bounds(de_sys_->is_interpolated);
          de_sys_->g_func_.update_interp_bounds(de_sys_->is_interpolated);


          break;
        } else {
          if (wkb) {
            if (maxindex_wkb <= 1) {
              if (nwkb1 > 1) {
                hnext = h * std::pow(1.0 / wkbdelta, 1.0 / (nwkb1 - 1));
              } else {
                hnext = 0.95 * h * 1.0 / wkbdelta;
              }
            } else {
              hnext = h * std::pow(1.0 / wkbdelta, 1.0 / (nwkb2 - 1));
            }
          } else {
            hnext = h * std::pow(1.0 / rkdelta, 1.0 / (nrk - 1));
          } 
          h = hnext;
          tnext = t + hnext;
          fnext = (h > 0) ? tf - tnext : tnext - tf;
        }
        totsteps++;
        // Checking for too many steps and low acceptance ratio:
        if (warn_once && totsteps % 5000 == 0) {
          warn_once = false;
          std::cerr << "Warning: the solver took " << totsteps
                    << " steps, and may take a while to converge." << std::endl;
          if (ssteps / totsteps < 0.05) {
            std::cerr << "Warning: the step acceptance ratio is below 5%, the "
                         "solver may take a while to converge."
                      << std::endl;
          }
        }

      }
    }

    // If integrating backwards, reverse dense output (because it will have been
    // reversed at the start)
    if ((de_sys_->is_interpolated && !de_sys_->w_func_.sign_) || (!de_sys_->is_interpolated && !sign)) {
        std::reverse(dotimes.begin(), dotimes.end());
        std::reverse(dosol.begin(), dosol.end());
        std::reverse(dodsol.begin(), dodsol.end());
    }

    // Write output to file if prompted
    if (!(*fo)) {
      std::string output(fo);
      std::ofstream f;
      f.open(output);
      f << "# Summary:\n# total steps taken: " + std::to_string(totsteps) +
               "\n# of which successful: " + std::to_string(ssteps) +
               "\n# of which" + +" wkb: " + std::to_string(wkbsteps) +
               "\n# time, sol, dsol, wkb? (type)\n";
      auto it_t = times.begin();
      auto it_w = wkbs.begin();
      auto it_x = sol.begin();
      auto it_dx = dsol.begin();
      for (int i = 0; i <= ssteps; i++) {
        f << std::setprecision(15) << *it_t << ";" << std::setprecision(15)
          << *it_x << ";" << std::setprecision(15) << *it_dx << ";" << *it_w
          << "\n";
        ++it_t;
        ++it_x;
        ++it_dx;
        ++it_w;
      }
      // print all dense output to file
      int dosize = dosol.size();
      auto it_dosol = dosol.begin();
      auto it_dotimes = dotimes.begin();
      auto it_dodsol = dodsol.begin();
      for (int i = 0; i < dosize; i++) {
        f << std::setprecision(20) << *it_dotimes << ";"
          << std::setprecision(20) << *it_dosol << ";" << *it_dodsol << ";\n";
        ++it_dosol;
        ++it_dodsol;
        ++it_dotimes;
      }

      f.close();
    }
  }

};

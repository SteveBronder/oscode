#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <oscode/interpolator.hpp>
#include <oscode/rksolver.hpp>
#include <oscode/system.hpp>
#include <oscode/wkbsolver.hpp>
#include <string>
#include <vector>

/** A class to store all information related to a numerical solution run.  */
class Solution {
private:
  /** A \a de_system object to carry information about the ODE. */
  de_system *de_sys_;
  double t, tf, rtol, atol, h0;
  std::complex<double> x, dx;
  int order;
  const char *fo;
  std::unique_ptr<WKBSolver> wkbsolver;

public:
  /** Object to call RK steps */
  RKSolver rksolver;

  /** Successful, total attempted, and successful WKB steps the solver took,
   * respectively  */
  int ssteps{0}, totsteps{0}, wkbsteps{0};
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
  // Experimental: list to contain continuous representation of the solution
  std::vector<Eigen::Matrix<std::complex<double>, 7, 1>> sol_vdm;

private:
  /** These define the event at which integration finishes (currently: when tf
   * is reached. */
  double fend, fnext;
  /** a boolean encoding the direction of integration: 1/True for forward. */
  bool sign{false};
  /** whether to produce dense output */
  bool dense_output{false};
  WKBSolver *set_wkb_solver_order(int order, de_system *&desys) {
    switch (order) {
    case 1:
      return new WKBSolver1(*de_sys_, order);
    case 2:
      return new WKBSolver2(*de_sys_, order);
    case 3:
      return new WKBSolver3(*de_sys_, order);
    };
    return new WKBSolver3(*de_sys_, order);
  }
public:
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
  Solution(de_system &de_sys, std::complex<double> x0, std::complex<double>
    dx0, double t_i, double t_f, int o=3, double r_tol=1e-4, double a_tol=0.0,
    double h_0=1, const char* full_output="") : de_sys_(&de_sys), t(t_i), tf(t_f), rtol(r_tol),
            atol(a_tol), h0(h_0),
            x(x0), dx(dx0), order(o), fo(full_output),
            wkbsolver(std::unique_ptr<WKBSolver>(set_wkb_solver_order(order, de_sys_))), 
            rksolver(RKSolver(*de_sys_)) {

    // Determine direction of integration, fend>0 and integration ends when
    // it crosses zero
    if ((t >= tf) && h0 < 0) {
      // backwards
      fend = t - tf;
      fnext = fend;
      de_sys_->Winterp_.sign_ = !de_sys_->is_interpolated_;
      de_sys_->Ginterp_.sign_ = !de_sys_->is_interpolated_;
      // sign is false by default
      //  sign = 0;

    } else if ((t <= tf) && h0 > 0) {
      // forward
      fend = tf - t;
      fnext = fend;
      de_sys_->Winterp_.sign_ = de_sys_->is_interpolated_;
      de_sys_->Ginterp_.sign_ = de_sys_->is_interpolated_;
      sign = !de_sys_->is_interpolated_;
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
    //dot_it = dotimes.end();
    dense_output = false;
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
  template <typename X = double>
  Solution(de_system &de_sys,
    std::complex<double> x0, std::complex<double> dx0, double t_i, double t_f,
    const X &do_times, int o=3, double r_tol=1e-4, double a_tol=0.0,
    double h_0=1, const char* full_output="") : de_sys_(&de_sys), t(t_i), tf(t_f), rtol(r_tol),
            atol(a_tol), h0(h_0),
            x(x0), dx(dx0), order(o), fo(full_output),
            wkbsolver(std::unique_ptr<WKBSolver>(set_wkb_solver_order(order, de_sys_))), 
            rksolver(RKSolver(*de_sys_)) {

    // Set parameters for solver
    // Determine direction of integration, fend>0 and integration ends when
    // it crosses zero
    if ((t >= tf) && h0 < 0) {
      // backwards
      fend = t - tf;
      fnext = fend;
      de_sys_->Winterp_.sign_ = !de_sys_->is_interpolated_;
      de_sys_->Ginterp_.sign_ = !de_sys_->is_interpolated_;
      // sign is false by default
    } else if ((t <= tf) && h0 > 0) {
      // forward
      fend = tf - t;
      fnext = fend;
        de_sys_->Winterp_.sign_ = de_sys_->is_interpolated_;
        de_sys_->Ginterp_.sign_ = de_sys_->is_interpolated_;
        sign = !de_sys_->is_interpolated_;
    } else {
      throw std::logic_error(
          "Direction of integration in conflict with direction of initial "
          "step, "
          "terminating. Please check your values for ti, tf, and h.");
      return;
    }

    // Dense output preprocessing: sort and reverse if necessary
    int dosize = do_times.size();
    dotimes.resize(dosize);
    dosol.resize(dosize);
    dodsol.resize(dosize);

    // Copy dense output points to list
    dotimes = do_times;

    // Reverse if necessary
    if ((de_sys_->is_interpolated_ && !de_sys_->Winterp_.sign_) ||
        (!de_sys_->is_interpolated_ && !sign)) {
      std::reverse(dotimes.begin(), dotimes.end());
    } else {
      std::sort(dotimes.begin(), dotimes.end());
    }

    //dot_it = dotimes.begin();
    dense_output = true;
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
    constexpr int nrk = 5;
    constexpr int nwkb1 = 2;
    constexpr int nwkb2 = 4;
    Eigen::Matrix<std::complex<double>, 1, 2> truncerr;
    Eigen::Matrix<double, 1, 2> errmeasure_rk;
    Eigen::Matrix<double, 1, 4> errmeasure_wkb;
    Eigen::Index maxindex_wkb{0}, maxindex_rk;
    double h = h0;
    double tnext = t + h;
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
    // NOTE: If any of these are resized and cause a memory allocation the program will segfault
    // since the iterator will be invalidated.
    auto it_dosol = dosol.begin();
    auto it_dodsol = dodsol.begin();
    auto dot_it = dotimes.begin();
    while (fend > 0) {
      // Check if we are reaching the end of integration
      if (fnext < 0) {
        h = tf - t;
        tnext = tf;
      }

      // Keep updating stepsize until step is accepted
      while (true) {
        // RK step
        Eigen::Matrix<std::complex<double>, 2, 2> rkstep = rksolver.step(x, dx, t, h);
        auto rkx = rkstep.col(0);
        auto rkerr = rkstep.col(1);
        // WKB step
        Eigen::Matrix<std::complex<double>, 3, 2> wkbstep = wkbsolver->step(dense_output, x, dx, t, h, rksolver.ws, rksolver.gs,
                                  rksolver.ws5, rksolver.gs5);
        auto wkbx = wkbstep.row(0);
        auto wkberr = wkbstep.row(2);
        truncerr = wkbstep.row(1);
        // Safety feature for when all wkb steps are 0 (truncer=0), but not
        // necessarily in good WKB regime:
        truncerr(0) = std::max(1e-10, abs(truncerr(0)));
        truncerr(1) = std::max(1e-10, abs(truncerr(1)));
        // dominant error calculation
        // Error scale measures
        errmeasure_rk(0) = std::abs(rkerr(0)) / (std::abs(rkx(0)) * rtol + atol);
        errmeasure_rk(1) = std::abs(rkerr(1)) / (std::abs(rkx(1)) * rtol + atol);
        errmeasure_wkb(0) = std::abs(truncerr(0)) / (std::abs(wkbx(0)) * rtol + atol),
        errmeasure_wkb(1) = std::abs(truncerr(1)) / (std::abs(wkbx(1)) * rtol + atol),
        errmeasure_wkb(2) = std::abs(wkberr(0)) / (std::abs(wkbx(0)) * rtol + atol),
        errmeasure_wkb(3) = std::abs(wkberr(1)) / (std::abs(wkbx(1)) * rtol + atol);
        double rkdelta = std::max(1e-10, errmeasure_rk.maxCoeff(&maxindex_rk));
        bool check_wkbdelta = std::isnan(errmeasure_wkb.maxCoeff()) ||
            std::isinf(std::real(wkbx(0))) ||
            std::isnan(std::real(wkbx(0))) ||
            std::isinf(std::imag(wkbx(0))) ||
            std::isnan(std::imag(wkbx(0))) ||
            std::isinf(std::real(wkbx(1))) ||
            std::isnan(std::real(wkbx(1))) ||
            std::isinf(std::imag(wkbx(1))) ||
            std::isnan(std::imag(wkbx(1)));
        double wkbdelta = check_wkbdelta ? std::numeric_limits<double>::infinity() : std::max(1e-10, errmeasure_wkb.maxCoeff(&maxindex_wkb));
        // predict next stepsize
        const double hrk = h * std::pow((1.0 / rkdelta), 1.0 / nrk);
        const double hwkb = h * std::pow(1.0 / wkbdelta, 1.0 / (maxindex_wkb <= 1 ? nwkb1 : nwkb2));
        // choose step with larger predicted stepsize
        const bool wkb = std::abs(hwkb) >= std::abs(hrk);
        std::complex<double> xnext = wkb ? wkbx(0) : rkx(0);
        std::complex<double> dxnext = wkb ? wkbx(1) : rkx(1);
        // if wkb step chosen, ignore truncation error in
        // stepsize-increase
        wkbdelta = wkb ? std::max(1e-10, errmeasure_wkb.tail(2).maxCoeff()) : wkbdelta;
        double hnext = wkb ? h * std::pow(1.0 / wkbdelta, 1.0 / nwkb2) : hrk;
        totsteps += 1;
        // Checking for too many steps and low acceptance ratio:
        if (totsteps > 5000) {
          std::cerr << "Warning: the solver took " << totsteps
                    << " steps, and may take a while to converge." << std::endl;
          if (ssteps / totsteps < 0.05) {
            std::cerr << "Warning: the step acceptance ratio is below 5%, the "
                         "solver may take a while to converge."
                      << std::endl;
          }
        }

        // check if chosen step was successful
        if (std::abs(hnext) >= std::abs(h)) {
          if (OSCODE_DEBUG) {
            std::cout << "All dense output points: " << std::endl;
          }
          if (dense_output) {
            //                    std::cout << *dot_it << std::endl;
            for (;(*dot_it - t >= 0 && tnext - *dot_it >= 0) ||
                   (*dot_it - t <= 0 && tnext - *dot_it <= 0); ++dot_it) {
              inner_dotimes.push_back(*dot_it);
            }
            if (inner_dotimes.size() > 0) {
              inner_dosols.resize(inner_dotimes.size());
              inner_dodsols.resize(inner_dotimes.size());
              if (wkb) {
                if (OSCODE_DEBUG) {
                  // Dense output after successful WKB step
                  std::cout << "Attempting " << inner_dosols.size() << " dense output points after successful WKB step from " << t << " to " << t+h << std::endl;
                }
                wkbsolver->dense_step(t, h, inner_dotimes, inner_dosols,
                                      inner_dodsols);
              } else {
                // Dense output after successful RK step
                for (auto it = inner_dotimes.begin(); it != inner_dotimes.end(); it++) {
                  rksolver.dense_step(t, h, x, dx, inner_dotimes, inner_dosols,
                                      inner_dodsols);
                }
              }
            }
          }
          for (auto inner_it = inner_dosols.begin(), inner_dit = inner_dodsols.begin();
           inner_it != inner_dosols.end() && it_dosol != dosol.end() &&
           inner_dit != inner_dodsols.end() &&
           it_dodsol != dodsol.end(); it_dodsol++, it_dosol++, inner_it++, inner_dit++) {
            *it_dosol = *inner_it;
            *it_dodsol = *inner_dit;
          }
          inner_dotimes.clear();
          inner_dosols.clear();
          inner_dodsols.clear();

          // record type of step
          wkbsteps += wkb;
          wkbs.push_back(wkb);
          xvdm = wkb ? wkbsolver->x_vdm : rksolver.x_vdm;
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
          de_sys_->Winterp_.update_interp_bounds(de_sys_->is_interpolated_);
          de_sys_->Ginterp_.update_interp_bounds(de_sys_->is_interpolated_);
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
      }
    }

    // If integrating backwards, reverse dense output (because it will have been
    // reversed at the start)
    if ((de_sys_->is_interpolated_ && !de_sys_->Winterp_.sign_) || (!de_sys_->is_interpolated_ && !sign)) {
        std::reverse(dotimes.begin(), dotimes.end());
        std::reverse(dosol.begin(), dosol.end());
        std::reverse(dodsol.begin(), dodsol.end());
    }

    // Write output to file if prompted
    if (!(*fo == 0)) {
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

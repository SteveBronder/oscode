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
  double t_, tf_, h0_;
  std::complex<double> x_, dx_;
  int order_;
  const char *fo;
  std::unique_ptr<WKBSolver> wkbsolver_;

public:
  /** Object to call RK steps */
  RKSolver rksolver_;

  /** Successful, total attempted, and successful WKB steps the solver took,
   * respectively  */
  int ssteps_{0}, totsteps_{0}, wkbsteps_{0};
  /** Lists to contain the solution and its derivative evaluated at internal
   * points taken by the solver (i.e. not dense output) after a run */
  std::vector<std::complex<double>> sol_, dsol_;
  /** List to contain the timepoints at which the solution and derivative are
   * internally evaluated by the solver */
  std::vector<double> times_;
  /** List to contain the "type" of each step (RK/WKB) taken internally by the
   * solver after a run */
  std::vector<bool> wkbs_;
  /** Lists to contain the timepoints at which dense output was evaluated. This
   * list will always be sorted in ascending order (with possible duplicates),
   * regardless of the order the timepoints were specified upon input. */
  std::vector<double> dotimes_;
  /** Lists to contain the dense output of the solution and its derivative */
  std::vector<std::complex<double>> dosol_, dodsol_;
  // Experimental: list to contain continuous representation of the solution
  std::vector<Eigen::Matrix<std::complex<double>, 7, 1>> sol_vdm_;

private:
  /** These define the event at which integration finishes (currently: when tf
   * is reached. */
  double fend_, fnext_;
  /** a boolean encoding the direction of integration: 1/True for forward. */
  bool sign_{false};
  /** whether to produce dense output */
  bool dense_output_{false};
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
  Solution(de_system &de_sys, std::complex<double> x0, std::complex<double> dx0,
           double t_i, double t_f, int o = 3, double h_0 = 1,
           const char *full_output = "")
      : de_sys_(&de_sys), t_(t_i), tf_(t_f), h0_(h_0), x_(x0), dx_(dx0),
        order_(o), fo(full_output), wkbsolver_(std::unique_ptr<WKBSolver>(
                                        set_wkb_solver_order(order_, de_sys_))),
        rksolver_(RKSolver(*de_sys_)) {

    // Determine direction of integration, fend_>0 and integration ends when
    // it crosses zero
    if ((t_ >= tf_) && h0_ < 0) {
      // backwards
      fend_ = t_ - tf_;
      fnext_ = fend_;
      de_sys_->Winterp_.sign_ = !de_sys_->is_interpolated_;
      de_sys_->Ginterp_.sign_ = !de_sys_->is_interpolated_;
      // sign is false by default
      //  sign = 0;

    } else if ((t_ <= tf_) && h0_ > 0) {
      // forward
      fend_ = tf_ - t_;
      fnext_ = fend_;
      de_sys_->Winterp_.sign_ = de_sys_->is_interpolated_;
      de_sys_->Ginterp_.sign_ = de_sys_->is_interpolated_;
      sign_ = !de_sys_->is_interpolated_;
    } else {
      throw std::logic_error(
          "Direction of integration in conflict with direction of initial "
          "step, "
          "terminating. Please check your values for ti, tf, and h.");
      return;
    }

    // No dense output desired if this constructor was called, so only output
    // answer at t_i and t_f
    dotimes_.push_back(t_i);
    dotimes_.push_back(t_f);
    dosol_.push_back(x0);
    dodsol_.push_back(dx0);
    // dot_it = dotimes_.end();
    dense_output_ = false;
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
  Solution(de_system &de_sys, std::complex<double> x0, std::complex<double> dx0,
           double t_i, double t_f, const X &do_times, int o = 3, double h_0 = 1,
           const char *full_output = "")
      : de_sys_(&de_sys), t_(t_i), tf_(t_f), h0_(h_0), x_(x0), dx_(dx0),
        order_(o), fo(full_output), wkbsolver_(std::unique_ptr<WKBSolver>(
                                        set_wkb_solver_order(order_, de_sys_))),
        rksolver_(RKSolver(*de_sys_)) {

    // Set parameters for solver
    // Determine direction of integration, fend_>0 and integration ends when
    // it crosses zero
    if ((t_ >= tf_) && h0_ < 0) {
      // backwards
      fend_ = t_ - tf_;
      fnext_ = fend_;
      de_sys_->Winterp_.sign_ = !de_sys_->is_interpolated_;
      de_sys_->Ginterp_.sign_ = !de_sys_->is_interpolated_;
      // sign is false by default
    } else if ((t_ <= tf_) && h0_ > 0) {
      // forward
      fend_ = tf_ - t_;
      fnext_ = fend_;
      de_sys_->Winterp_.sign_ = de_sys_->is_interpolated_;
      de_sys_->Ginterp_.sign_ = de_sys_->is_interpolated_;
      sign_ = !de_sys_->is_interpolated_;
    } else {
      throw std::logic_error(
          "Direction of integration in conflict with direction of initial "
          "step, "
          "terminating. Please check your values for ti, tf, and h.");
      return;
    }

    // Dense output preprocessing: sort and reverse if necessary
    int dosize = do_times.size();
    dotimes_.resize(dosize);
    dosol_.resize(dosize);
    dodsol_.resize(dosize);

    // Copy dense output points to list
    dotimes_ = do_times;

    // Reverse if necessary
    if ((de_sys_->is_interpolated_ && !de_sys_->Winterp_.sign_) ||
        (!de_sys_->is_interpolated_ && !sign_)) {
      std::reverse(dotimes_.begin(), dotimes_.end());
    } else {
      std::sort(dotimes_.begin(), dotimes_.end());
    }

    // dot_it = dotimes_.begin();
    dense_output_ = true;
  }
  /** \brief Function to solve the ODE \f$ \ddot{x} + 2\gamma(t)\dot{x} +
   * \omega^2(t)x = 0 \f$ for \f$ x(t), \frac{dx}{dt} \f$.
   *
   * While solving the ODE, this function will populate the \a Solution object
   * with the following results:
   *
   */
  void solve(double rtol, double atol) {

    // Settings for MS
    constexpr int nrk = 5;
    constexpr int nwkb1 = 2;
    constexpr int nwkb2 = 4;
    Eigen::Matrix<std::complex<double>, 1, 2> truncerr;
    Eigen::Matrix<double, 1, 2> errmeasure_rk;
    Eigen::Matrix<double, 2, 2> errmeasure_wkb;
    Eigen::Index maxindex_wkb{0}, maxindex_rk;
    double h = h0_;
    double tnext = t_ + h;
    // Initialise stats
    sol_.push_back(x_);
    dsol_.push_back(dx_);
    times_.push_back(t_);
    wkbs_.push_back(false);
    ssteps_ = 0;
    totsteps_ = 0;
    wkbsteps_ = 0;
    // Dense output
    std::vector<double> inner_dotimes;
    std::vector<std::complex<double>> inner_dosols, inner_dodsols;
    Eigen::Matrix<std::complex<double>, 1, 2> y_dense_rk;
    // Experimental continuous solution, vandermonde representation
    Eigen::Matrix<std::complex<double>, 7, 1> xvdm;
    std::size_t do_solve_count = 0;
    std::size_t do_dodsol_count = 0;
    while (fend_ > 0) {
      // Check if we are reaching the end of integration
      if (fnext_ < 0) {
        h = tf_ - t_;
        tnext = tf_;
      }

      // Keep updating stepsize until step is accepted
      while (true) {
        // RK step
        Eigen::Matrix<std::complex<double>, 2, 2> rkstep =
            rksolver_.step(x_, dx_, t_, h);
        auto rkx = rkstep.col(0);
        auto rkerr = rkstep.col(1);
        // WKB step
        Eigen::Matrix<std::complex<double>, 3, 2> wkbstep =
            wkbsolver_->step(dense_output_, x_, dx_, t_, h, rksolver_.ws_,
                             rksolver_.gs_, rksolver_.ws5_, rksolver_.gs5_);
        auto wkbx = wkbstep.row(0);
        auto wkberr = wkbstep.row(2);
        truncerr = wkbstep.row(1);
        // Safety feature for when all wkb steps are 0 (truncer=0), but not
        // necessarily in good WKB regime:
        truncerr(0) = std::max(1e-10, abs(truncerr(0)));
        truncerr(1) = std::max(1e-10, abs(truncerr(1)));
        // dominant error calculation
        // Error scale measures
        const auto tols = rtol + atol;
        errmeasure_rk.array() =
            rkerr.array().abs() / (rkx.array().abs() * tols);
        errmeasure_wkb.col(0).array() =
            truncerr.array().abs() / (wkbx.array().abs() * tols);
        errmeasure_wkb.col(1).array() =
            wkberr.array().abs() / (wkbx.array().abs() * tols);
        double rkdelta = std::max(1e-10, errmeasure_rk.maxCoeff(&maxindex_rk));
        const bool check_wkbdelta =
            std::isnan(errmeasure_wkb.reshaped().array().maxCoeff()) ||
            std::isinf(std::real(wkbx(0))) || std::isnan(std::real(wkbx(0))) ||
            std::isinf(std::imag(wkbx(0))) || std::isnan(std::imag(wkbx(0))) ||
            std::isinf(std::real(wkbx(1))) || std::isnan(std::real(wkbx(1))) ||
            std::isinf(std::imag(wkbx(1))) || std::isnan(std::imag(wkbx(1)));
        double wkbdelta =
            check_wkbdelta
                ? std::numeric_limits<double>::infinity()
                : std::max(1e-10, errmeasure_wkb.reshaped().array().maxCoeff(
                                      &maxindex_wkb));
        // predict next stepsize
        const double hrk = h * std::pow((1.0 / rkdelta), 1.0 / nrk);
        const double hwkb =
            h *
            std::pow(1.0 / wkbdelta, 1.0 / (maxindex_wkb <= 1 ? nwkb1 : nwkb2));
        // choose step with larger predicted stepsize
        const bool wkb = std::abs(hwkb) >= std::abs(hrk);
        if (wkb) {
          std::complex<double> xnext = wkbx(0);
          std::complex<double> dxnext = wkbx(1);
          // if wkb step chosen, ignore truncation error in
          // stepsize-increase
          wkbdelta = std::max(1e-10, errmeasure_wkb.col(1).maxCoeff());
          double hnext = h * std::pow(1.0 / wkbdelta, 1.0 / nwkb2);
          totsteps_ += 1;
          // Checking for too many steps and low acceptance ratio:
          if (totsteps_ > 5000) {
            std::cerr << "Warning: the solver took " << totsteps_
                      << " steps, and may take a while to converge."
                      << std::endl;
            if (ssteps_ / totsteps_ < 0.05) {
              std::cerr
                  << "Warning: the step acceptance ratio is below 5%, the "
                     "solver may take a while to converge."
                  << std::endl;
            }
          }

          // check if chosen step was successful
          if (std::abs(hnext) >= std::abs(h)) {
            if (OSCODE_DEBUG) {
              std::cout << "All dense output points: " << std::endl;
            }
            if (dense_output_) {
              for (auto dot_it = dotimes_.begin() + do_solve_count;
                   (*dot_it - t_ >= 0 && tnext - *dot_it >= 0) ||
                   (*dot_it - t_ <= 0 && tnext - *dot_it <= 0);
                   ++dot_it) {
                inner_dotimes.push_back(*dot_it);
                do_solve_count++;
              }
              if (inner_dotimes.size() > 0) {
                inner_dosols.resize(inner_dotimes.size());
                inner_dodsols.resize(inner_dotimes.size());
                if (OSCODE_DEBUG) {
                  // Dense output after successful WKB step
                  std::cout
                      << "Attempting " << inner_dosols.size()
                      << " dense output points after successful WKB step from "
                      << t_ << " to " << t_ + h << std::endl;
                }
                wkbsolver_->dense_step(t_, h, inner_dotimes, inner_dosols,
                                       inner_dodsols);
              }
            }
            for (auto inner_it = inner_dosols.begin(),
                      inner_dit = inner_dodsols.begin(),
                      it_dosol = dosol_.begin() + do_dodsol_count,
                      it_dodsol = dodsol_.begin() + do_dodsol_count;
                 inner_it != inner_dosols.end() && it_dosol != dosol_.end() &&
                 inner_dit != inner_dodsols.end() && it_dodsol != dodsol_.end();
                 it_dodsol++, it_dosol++, inner_it++, inner_dit++,
                      do_dodsol_count++) {
              *it_dosol = std::move(*inner_it);
              *it_dodsol = std::move(*inner_dit);
            }
            inner_dotimes.clear();
            inner_dosols.clear();
            inner_dodsols.clear();

            // record type of step
            wkbsteps_++;
            wkbs_.push_back(true);
            xvdm = wkbsolver_->x_vdm;
            sol_.push_back(xnext);
            dsol_.push_back(dxnext);
            sol_vdm_.push_back(xvdm);
            times_.push_back(tnext);
            tnext += hnext;
            x_ = xnext;
            dx_ = dxnext;
            t_ += h;
            h = hnext;
            fend_ = h > 0 ? tf_ - t_ : t_ - tf_;
            fnext_ = h > 0 ? tf_ - tnext : tnext - tf_;
            ssteps_ += 1;
            // Update interpolation bounds
            de_sys_->Winterp_.update_interp_bounds(de_sys_->is_interpolated_);
            de_sys_->Ginterp_.update_interp_bounds(de_sys_->is_interpolated_);
            break;
          } else {
            if (maxindex_wkb <= 1) {
              if (nwkb1 > 1) {
                hnext = h * std::pow(1.0 / wkbdelta, 1.0 / (nwkb1 - 1));
              } else {
                hnext = 0.95 * h * 1.0 / wkbdelta;
              }
            } else {
              hnext = h * std::pow(1.0 / wkbdelta, 1.0 / (nwkb2 - 1));
            }
            h = hnext;
            tnext = t_ + hnext;
            fnext_ = (h > 0) ? tf_ - tnext : tnext - tf_;
          }

        } else {
          std::complex<double> xnext = rkx(0);
          std::complex<double> dxnext = rkx(1);
          double hnext = hrk;
          totsteps_ += 1;
          // Checking for too many steps and low acceptance ratio:
          if (totsteps_ > 5000) {
            std::cerr << "Warning: the solver took " << totsteps_
                      << " steps, and may take a while to converge."
                      << std::endl;
            if (ssteps_ / totsteps_ < 0.05) {
              std::cerr
                  << "Warning: the step acceptance ratio is below 5%, the "
                     "solver may take a while to converge."
                  << std::endl;
            }
          }

          // check if chosen step was successful
          if (std::abs(hnext) >= std::abs(h)) {
            if (OSCODE_DEBUG) {
              std::cout << "All dense output points: " << std::endl;
            }
            if (dense_output_) {
              //                    std::cout << *dot_it << std::endl;
              for (auto dot_it = dotimes_.begin() + do_solve_count;
                   (*dot_it - t_ >= 0 && tnext - *dot_it >= 0) ||
                   (*dot_it - t_ <= 0 && tnext - *dot_it <= 0);
                   ++dot_it) {
                inner_dotimes.push_back(*dot_it);
                do_solve_count++;
              }
              if (inner_dotimes.size() > 0) {
                inner_dosols.resize(inner_dotimes.size());
                inner_dodsols.resize(inner_dotimes.size());
                // Dense output after successful RK step
                for (auto it = inner_dotimes.begin(); it != inner_dotimes.end();
                     it++) {
                  rksolver_.dense_step(t_, h, x_, dx_, inner_dotimes,
                                       inner_dosols, inner_dodsols);
                }
              }
            }
            for (auto inner_it = inner_dosols.begin(),
                      inner_dit = inner_dodsols.begin(),
                      it_dosol = dosol_.begin() + do_dodsol_count,
                      it_dodsol = dodsol_.begin() + do_dodsol_count;
                 inner_it != inner_dosols.end() && it_dosol != dosol_.end() &&
                 inner_dit != inner_dodsols.end() && it_dodsol != dodsol_.end();
                 it_dodsol++, it_dosol++, inner_it++, inner_dit++,
                      do_dodsol_count++) {
              *it_dosol = std::move(*inner_it);
              *it_dodsol = std::move(*inner_dit);
            }
            inner_dotimes.clear();
            inner_dosols.clear();
            inner_dodsols.clear();

            // record type of step
            wkbs_.push_back(false);
            xvdm = rksolver_.x_vdm_;
            sol_.push_back(xnext);
            dsol_.push_back(dxnext);
            sol_vdm_.push_back(xvdm);
            times_.push_back(tnext);
            tnext += hnext;
            x_ = xnext;
            dx_ = dxnext;
            t_ += h;
            h = hnext;
            fend_ = h > 0 ? tf_ - t_ : t_ - tf_;
            fnext_ = h > 0 ? tf_ - tnext : tnext - tf_;
            ssteps_ += 1;
            // Update interpolation bounds
            de_sys_->Winterp_.update_interp_bounds(de_sys_->is_interpolated_);
            de_sys_->Ginterp_.update_interp_bounds(de_sys_->is_interpolated_);
            break;
          } else {
            hnext = h * std::pow(1.0 / rkdelta, 1.0 / (nrk - 1));
            h = hnext;
            tnext = t_ + hnext;
            fnext_ = (h > 0) ? tf_ - tnext : tnext - tf_;
          }
        }
      }
    }

    // If integrating backwards, reverse dense output (because it will have been
    // reversed at the start)
    if ((de_sys_->is_interpolated_ && !de_sys_->Winterp_.sign_) ||
        (!de_sys_->is_interpolated_ && !sign_)) {
      std::reverse(dotimes_.begin(), dotimes_.end());
      std::reverse(dosol_.begin(), dosol_.end());
      std::reverse(dodsol_.begin(), dodsol_.end());
    }

    // Write output to file if prompted
    if (!(*fo == 0)) {
      std::string output(fo);
      std::ofstream f;
      f.open(output);
      f << "# Summary:\n# total steps taken: " + std::to_string(totsteps_) +
               "\n# of which successful: " + std::to_string(ssteps_) +
               "\n# of which" + +" wkb: " + std::to_string(wkbsteps_) +
               "\n# time, sol, dsol, wkb? (type)\n";
      auto it_t = times_.begin();
      auto it_w = wkbs_.begin();
      auto it_x = sol_.begin();
      auto it_dx = dsol_.begin();
      for (int i = 0; i <= ssteps_; i++) {
        f << std::setprecision(15) << *it_t << ";" << std::setprecision(15)
          << *it_x << ";" << std::setprecision(15) << *it_dx << ";" << *it_w
          << "\n";
        ++it_t;
        ++it_x;
        ++it_dx;
        ++it_w;
      }
      // print all dense output to file
      int dosize = dosol_.size();
      auto it_dosol = dosol_.begin();
      auto it_dotimes = dotimes_.begin();
      auto it_dodsol = dodsol_.begin();
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

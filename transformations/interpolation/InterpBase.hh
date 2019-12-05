#pragma once

#include <string>
#include "InSegment.hh"
#include "InterpStrategy.hh"

/**
 * @brief Basic Interpolation (unordered).
 *
 * For a given `x`, `y` and `newx` computes `newy` via user-provided function this->interpolation_formula:
 *
 * The object inherits InSegment that determines the segments to be used for interpolation.
 *
 * @note No check is performed on the value of `b`.
 *
 * Inputs:
 *   - interp.newx -- array with points with fine steps to interpolate on.
 *   - interp.x -- array with coarse points.
 *   - interp.y -- values of a function on `x`.
 *   - interp.insegment -- segments of `newx` in `x`. See InSegment.
 *
 * Outputs:
 *   - interp.interp
 *
 * The connection may be done via InterpBase::interpolate() method.
 *
 * @author Maxim Gonchar
 * @date 02.2017
 */

class InterpBase: public InSegment,
                  public TransformationBind<InterpBase> {
public:
  using TransformationBind<InterpBase>::transformation_;

  InterpBase();                                                                                                ///< Constructor.
  InterpBase(SingleOutput& x, SingleOutput& newx);                                                             ///< Constructor.
  InterpBase(SingleOutput& x, SingleOutput& y, SingleOutput& newx);                                            ///< Constructor.
  virtual ~InterpBase() {};

  TransformationDescriptor add_transformation(const std::string& name="");
  void bind_transformations();
  void bind_inputs();
  void set_underflow_strategy(GNA::Interpolation::Strategy strategy) noexcept;
  void set_overflow_strategy(GNA::Interpolation::Strategy strategy) noexcept;
  void set_fill_value(double x) noexcept {m_fill_value = x;};
  void set(SingleOutput& x, SingleOutput& newx);

  OutputDescriptor interpolate(SingleOutput& x, SingleOutput& y, SingleOutput& newx);                        ///< Initialize transformations by connecting `x`, `y` and `newx` outputs.

  void do_interpolate(FunctionArgs& fargs);                                                                    ///< Do the interpolation.
private:

  virtual double interpolation_formula(double x, double y, double k, double point) const noexcept = 0;                      ///< Abstract interface for initializing expression used for interploation. Must be provided by the derived class.
  virtual Eigen::ArrayXd compute_weights(const Eigen::ArrayXd& xs, const Eigen::ArrayXd& ys, const Eigen::ArrayXd& widths, size_t nseg) const noexcept = 0;      ///< Abstract interface for computing the weights for Interpolation formula. Must be provided by the derived class.
  GNA::Interpolation::Strategy m_underflow_strategy{GNA::Interpolation::Strategy::Extrapolate};                                                                 ///< Strategy to use for underflow points.
  GNA::Interpolation::Strategy m_overflow_strategy{GNA::Interpolation::Strategy::Extrapolate};                                                                  ///< Strategy to use for overflow points.
  double m_fill_value{0.};
};

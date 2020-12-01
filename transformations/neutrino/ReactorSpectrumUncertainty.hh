#pragma once

#include <string>
#include "InSegment.hh"

/**
 * @brief Bin-wise product necesserary for implementation of correlated and
 * uncorrelated reactor uncertainties in Huber-Mueller model
 *
 * The object inherits InSegment that determines the segments to be used for in-bin-product.
 *
 * TODO: write-up documentation
 * @author Konstantin Treskov
 * @date 10.2020
 */

class ReactorSpectrumUncertainty: public InSegment,
                    public TransformationBind<ReactorSpectrumUncertainty> {
public:
  using TransformationBind<ReactorSpectrumUncertainty>::transformation_;

  ReactorSpectrumUncertainty();                                                                                                ///< Constructor.
  ReactorSpectrumUncertainty(OutputDescriptor& bins, OutputDescriptor& bin_weights);                                                   ///< Constructor.
  virtual ~ReactorSpectrumUncertainty() {};

  void in_bin_product(FunctionArgs& fargs);

  enum class OutlierStrategy {Ignore, PropagateClosest, RaiseException};
  OutlierStrategy m_strategy = OutlierStrategy::PropagateClosest;
  void set_strategy(ReactorSpectrumUncertainty::OutlierStrategy strategy) noexcept {
      m_strategy = strategy;
  }
};

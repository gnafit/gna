#include "ReactorSpectrumUncertainty.hh"
#include "TypesFunctions.hh"
#include "TypeClasses.hh"
#include "fmt/format.h"

#include <optional>
#include <stdexcept>

using std::next;
using std::prev;
using std::advance;
using namespace std::literals;

ReactorSpectrumUncertainty::ReactorSpectrumUncertainty() : InSegment() {
  transformation_("in_bin_product")
    .input("insegment")        /// 0
    .input("correction")       /// 1
    .input("spectrum")         /// 2
    .output("corrected_spectrum")
    .types(TypesFunctions::pass<2,0>)
    /* .types(TypesFunctions::ifPoints<0>, TypesFunctions::ifPoints<1>, TypesFunctions::ifPoints<2>) */
    /* .types(TypesFunctions::ifSame2<0,2>) */
    .types(TypesFunctions::if1d<0>, TypesFunctions::if1d<1>)
    .func(&ReactorSpectrumUncertainty::in_bin_product)
    ;
}

ReactorSpectrumUncertainty::ReactorSpectrumUncertainty(OutputDescriptor& bins, OutputDescriptor& correction) : ReactorSpectrumUncertainty()
{
    auto insegment = transformations.front();
    auto& seg_inputs = insegment.inputs;
    bins >> seg_inputs[1];

    auto product = transformations.back();
    auto& prod_inps = product.inputs;
    insegment.outputs[0] >> prod_inps[0];
    correction >> prod_inps[1];
}

void ReactorSpectrumUncertainty::in_bin_product(FunctionArgs& fargs){
  const auto& args = fargs.args;                                                  /// name inputs
  auto& rets = fargs.rets;                                                        /// name outputs

  const auto* insegment_buf = args[0].buffer;                                    /// insegment buffer
  const auto& correction = args[1].x;                                             /// correction to spectrum
  const auto& spectrum = args[2].x;                                               /// antineutrino spectrum

  const auto  npoints = spectrum.size();                                          /// number of points
  using SizeType = std::remove_cv_t<decltype(npoints)>;

  const auto  nseg = correction.size();                                            /// number of segments, maximal coeff is overflow


  auto result = rets[0].buffer;                                                 /// output buffer write buffer

  const auto* correction_buf = args[1].buffer;                                  /// correction buffer
  const auto* spectrum_buf   = args[2].buffer;                                  /// spectrum buffer


  std::optional<SizeType> cur_segment = std::nullopt;
  for(SizeType i{0}; i<npoints; ++i){
    auto seg = *insegment_buf;
    if (!cur_segment) {
        cur_segment = seg;
    }
    if (cur_segment != seg) {
        cur_segment = seg;
        if (cur_segment < nseg && cur_segment>=0) {                                  /// no increment for overflow bin
            advance(correction_buf, 1);
        }
    }

    const auto spectrum_val = *spectrum_buf;
    const auto correction_val = *correction_buf;

    if( seg<0 ) {                                                  /// underflow
      switch (m_strategy) {
          case (OutlierStrategy::Ignore):
              *result = spectrum_val;
              break;
          case (OutlierStrategy::PropagateClosest):
              *result = spectrum_val*correction_val ;
              break;
          case (OutlierStrategy::RaiseException):
              auto msg = "Don't know how to apply correction for undeflow bins"s;
              throw std::runtime_error(msg);
              break;
      }
    }
    else if( seg>=nseg ) {                                         /// overflow
      switch (m_strategy) {
          case (OutlierStrategy::Ignore):
              *result = spectrum_val;
              break;
          case (OutlierStrategy::PropagateClosest):
              *result = spectrum_val*correction_val ;
              break;
          case (OutlierStrategy::RaiseException):
              auto msg = "Don't know how to apply correction for overflow bins"s;
              throw std::runtime_error(msg);
              break;
      }
    } else {                                                             /// plain correction
        *result = spectrum_val*correction_val;
    }

    advance(spectrum_buf, 1);
    advance(result, 1);
    }
}

#include "InterpBase.hh"
#include "TypesFunctions.hh"
#include "fmt/format.h"

#include <stdexcept>

using std::next;
using std::prev;
using std::advance;
using namespace GNA::Interpolation;

InterpBase::InterpBase() : InSegment() {
  add_transformation();
  add_input();
  set_open_input();
}

InterpBase::InterpBase(SingleOutput& x, SingleOutput& newx) : InterpBase()
{
  set(x, newx);
  bind_inputs();
}

InterpBase::InterpBase(SingleOutput& x, SingleOutput& y, SingleOutput& newx) : InterpBase()
{
  interpolate(x, y, newx);
}

TransformationDescriptor InterpBase::add_transformation(const std::string& name){
  transformation_(new_transformation_name(name))
    .input("newx")             /// 0
    .input("x")                /// 1
    .input("insegment")        /// 2
    .input("widths")           /// 3
    .types(TypesFunctions::ifPoints<0>)                                     /// newx is an array of any shape
    .types(TypesFunctions::ifPoints<1>, TypesFunctions::if1d<1>)            /// x is an 1d array
    .types(TypesFunctions::ifPoints<2>, TypesFunctions::ifSameShape2<0,2>)  /// segment index is of shape of newx
    .types(TypesFunctions::ifPoints<3>, TypesFunctions::if1d<3>)            /// widths is an 1d array
    .types(TypesFunctions::ifSame2<1,4>, TypesFunctions::ifBinsEdges<3,1>)
    .types(TypesFunctions::ifPoints<4>, TypesFunctions::if1d<4>)            /// y is an 1d array
    .types(TypesFunctions::ifSameInRange<4,-1,true>, TypesFunctions::passToRange<0,0,-1,true>)
    .func(&InterpBase::do_interpolate)
    ;

  reset_open_input();
  bind_transformations();
  return transformations.back();
}

void InterpBase::set(SingleOutput& x, SingleOutput& newx){
  auto segments = transformations.front();
  auto sinputs  = segments.inputs;
  sinputs[0].connect(newx.single());
  sinputs[1].connect(x.single());
}

void InterpBase::bind_transformations(){
  auto segments = transformations.front();
  auto interp = transformations.back();

  auto& outputs = segments.outputs;
  auto& inputs = interp.inputs;

  outputs[0] >> inputs[2];
  outputs[1] >> inputs[3];
}

void InterpBase::bind_inputs(){
  auto segments = transformations.front();
  auto interp = transformations.back();

  auto& seg_inputs = segments.inputs;
  auto& inputs = interp.inputs;

  seg_inputs[0].output() >> inputs[0];
  seg_inputs[1].output() >> inputs[1];
}

OutputDescriptor InterpBase::interpolate(SingleOutput& x, SingleOutput& y, SingleOutput& newx){
  set(x, newx);
  auto output = add_input(y);
  bind_inputs();
  return output;
}

OutputDescriptor InterpBase::setXY(SingleOutput& x, SingleOutput& y){
  auto segments = transformations.front();
  auto sinputs  = segments.inputs;
  auto interp = transformations.back();
  auto& inputs = interp.inputs;
  x.single() >> sinputs[1];
  x.single() >> inputs[1];

  auto output = add_input(y);
  return output;
}

void InterpBase::do_interpolate(FunctionArgs& fargs){
  const auto& args = fargs.args;                                                  /// name inputs
  auto& rets = fargs.rets;                                                        /// name outputs

  const auto& points_a = args[0].x;                                               /// new x points
  const auto  npoints = points_a.size();                                          /// number of points
  const auto& x_a = args[1].x;                                                    /// x of segments
  const auto& widths_a = args[3].x;                                               /// segment widths

  const auto nseg = x_a.size()-1;                                                 /// number of segments

  const auto insegment_start = args[2].buffer;                                    /// insegment buffer

  for (size_t ret = 0; ret < rets.size(); ++ret) {
    auto insegment = insegment_start;                                       /// insegment buffer
    auto point = points_a.data();                                           /// point read buffer
    const auto x_buffer = x_a.data();                                       /// x's buffer

    auto& y_a = args[4+ret].x;                                              /// y of segments, the offset
    const auto  y_buffer = y_a.data();                                      /// y's buffer

    const auto  k_a = compute_weights(x_a, y_a, widths_a, nseg);            /// k coefficient (weights)
    const auto  k_buffer = k_a.data();                                      /// k buffer

    auto result = rets[ret].buffer;                                         /// interpolation write buffer
    using SizeType = std::remove_cv_t<decltype(npoints)>;
    for(SizeType i{0}; i<npoints; ++i){
      auto idx = static_cast<size_t>(*insegment);
      if( *insegment<0 ) {                                                  /// underflow
        switch (m_underflow_strategy) {
            case (Strategy::Constant):
                *result = m_fill_value;
                break;
            case (Strategy::Extrapolate):
                idx = 0u;
                *result = interpolation_formula_below(x_buffer[idx],  y_buffer[idx], k_buffer[idx], *point);
                break;
        }
      }
      else if( *insegment>=nseg ) {                                         /// overflow
        switch (m_overflow_strategy) {
            case (Strategy::Constant):
                *result = m_fill_value;
                break;
            case (Strategy::Extrapolate):
                idx = nseg-1u;
                *result = interpolation_formula_above(x_buffer[idx],  y_buffer[idx], k_buffer[idx], *point);
                break;
        }
      } else {                                                              /// interpolation in definition range
       *result = interpolation_formula(x_buffer[idx],  y_buffer[idx], k_buffer[idx], *point);
      }

      advance(point, 1);
      advance(result, 1);
      advance(insegment, 1);
    }
  }
}

void InterpBase::set_underflow_strategy(GNA::Interpolation::Strategy strategy) noexcept {
    m_underflow_strategy = strategy;
}

void InterpBase::set_overflow_strategy(GNA::Interpolation::Strategy strategy) noexcept {
    m_overflow_strategy = strategy;
}

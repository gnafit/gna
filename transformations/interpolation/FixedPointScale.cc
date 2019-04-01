#include "FixedPointScale.hh"
#include "TypesFunctions.hh"
#include "fmt/format.h"

#include <TMath.h>
#include <stdexcept>

using std::next;
using std::prev;
using std::advance;
using TMath::Exp;

FixedPointScale::FixedPointScale() : InSegment() {
  add_transformation();
  add_input();
  set_open_input();

  //if(underflow_strategy.length()){
    //this->setUnderflowStrategy(underflow_strategy);
  //}
  //if(overflow_strategy.length()){
    //this->setOverflowStrategy(overflow_strategy);
  //}
}

FixedPointScale::FixedPointScale(SingleOutput& x, SingleOutput& fixedpoint) : FixedPointScale()
{
  set(x, fixedpoint);
  bind_inputs();
}

FixedPointScale::FixedPointScale(SingleOutput& x, SingleOutput& y, SingleOutput& fixedpoint) : FixedPointScale()
{
  scale(x, y, fixedpoint);
}

TransformationDescriptor FixedPointScale::add_transformation(const std::string& name){
  transformation_(new_transformation_name(name))
    .input("fixedpoint")       /// 0
    .input("x")                /// 1
    .input("insegment")        /// 2
    .input("widths")           /// 3
    .types(TypesFunctions::ifPoints<0>, TypesFunctions::if1d<0>)            /// fixedpoint is 1d array
    .types(TypesFunctions::ifPoints<1>, TypesFunctions::if1d<1>)            /// x is an 1d array
    .types(TypesFunctions::ifPoints<2>, TypesFunctions::ifSameShape2<0,2>)  /// segment index is of shape of fixedpoint
    .types(TypesFunctions::ifPoints<3>, TypesFunctions::if1d<3>)            /// widths is an 1d array
    .types(TypesFunctions::ifSame2<1,4>, TypesFunctions::ifBinsEdges<3,1>)
    .types(TypesFunctions::ifPoints<4>, TypesFunctions::if1d<4>)            /// y is an 1d array
    .types(TypesFunctions::ifSameInRange<4,-1,true>, TypesFunctions::passToRange<1,0,-1,true>)
    .types(&FixedPointScale::check)
    .func(&FixedPointScale::do_scale)
    ;

  reset_open_input();
  bind_transformations();
  return transformations.back();
}

void FixedPointScale::set(SingleOutput& x, SingleOutput& fixedpoint){
  auto segments = transformations.front();
  auto sinputs  = segments.inputs;
  sinputs[0].connect(fixedpoint.single());
  sinputs[1].connect(x.single());
}

void FixedPointScale::bind_transformations(){
  auto segments=transformations.front();
  auto interp=transformations.back();

  auto& outputs=segments.outputs;
  auto& inputs=interp.inputs;

  outputs[0]>>inputs[2];
  outputs[1]>>inputs[3];
}

void FixedPointScale::bind_inputs(){
  auto segments=transformations.front();
  auto interp=transformations.back();

  auto& seg_inputs=segments.inputs;
  auto& inputs=interp.inputs;

  seg_inputs[0].output()>>inputs[0];
  seg_inputs[1].output()>>inputs[1];
}

OutputDescriptor FixedPointScale::scale(SingleOutput& x, SingleOutput& y, SingleOutput& fixedpoint){
  set(x, fixedpoint);
  auto output=add_input(y);
  bind_inputs();
  return output;
}

void FixedPointScale::check(TypesFunctionArgs& fargs){
  auto& dt=fargs.args[0];
  if(dt.shape[0]!=1){
    fargs.args.error(dt, "Fixed point should be of size 1");
  }
}

void FixedPointScale::do_scale(FunctionArgs& fargs){
  auto& args=fargs.args;                                                  /// name inputs
  auto& rets=fargs.rets;                                                  /// name outputs

  auto& x_a=args[1].x;                                                    /// x of segments
  auto nseg=x_a.size()-1;                                                 /// number of segments

  auto insegment=args[2].x(0);                                            /// fixed point index
  if(insegment<0 || insegment>=nseg){
    throw std::runtime_error("Invalid fixed point position");
  }
  auto idx = static_cast<size_t>(insegment);
  auto fixed_point=args[0].x(0);                                          /// fixed point
  auto x1=x_a(idx);
  //auto x2=x_a(idx+1);
  auto width=args[3].x(idx);                                              /// segment width

  for (size_t ret = 0; ret < rets.size(); ++ret) {
    auto& y_a=args[4+ret].x;                                              /// y of segments, the offset
    auto y1=y_a(idx);
    auto y2=y_a(idx+1);
    auto  k=((y2-y1)/width);                                              /// k coefficient

    auto scale = k*(fixed_point - x1) + y1;
    rets[ret].x = fixed_point*y_a/scale;                                  /// scale
  }
}


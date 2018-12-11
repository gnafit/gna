#include "InterpExpo.hh"
#include "TypesFunctions.hh"

#include <TMath.h>
#include <stdexcept>

using std::next;
using std::prev;
using std::advance;
using TMath::Exp;

//InterpExpo::InterpExpo(const std::string& underflow_strategy, const std::string& overflow_strategy) : InSegment() {

InterpExpo::InterpExpo() : InSegment() {
  transformation_("interp")
    .input("x")                /// 0
    .input("y")                /// 1
    .input("insegment")        /// 2
    .input("widths")           /// 3
    .input("newx")             /// 4
    .output("interp")          /// 0
    .types(TypesFunctions::ifPoints<4>)                                    /// newx is an array of any shape
    .types(TypesFunctions::ifPoints<0>, TypesFunctions::if1d<0>)           /// x is an 1d array
    .types(TypesFunctions::ifPoints<1>, TypesFunctions::if1d<1>)           /// y is an 1d array
    .types(TypesFunctions::ifPoints<2>, TypesFunctions::ifSameShape2<4,2>) /// segment index is of shape of newx
    .types(TypesFunctions::ifPoints<3>, TypesFunctions::if1d<3>)           /// widths is an 1d array
    .types(TypesFunctions::ifSame2<0,1>, TypesFunctions::ifBinsEdges<3,0>)
    .types(TypesFunctions::pass<4,0>)
    .func(&InterpExpo::do_interpolate)
    ;

  //if(underflow_strategy.length()){
    //this->setUnderflowStrategy(underflow_strategy);
  //}
  //if(overflow_strategy.length()){
    //this->setOverflowStrategy(overflow_strategy);
  //}
}

void InterpExpo::interpolate(SingleOutput& x, SingleOutput& y, SingleOutput& newx){
  auto segments = this->t_["insegment"];
  auto sinputs  = segments.inputs();
  auto soutputs = segments.outputs();
  sinputs[0].connect(newx.single());
  sinputs[1].connect(x.single());

  auto interp = this->t_["interp"];
  auto iinputs = interp.inputs();
  iinputs[0].connect(x.single());
  iinputs[1].connect(y.single());
  iinputs[2].connect(soutputs[0]);
  iinputs[3].connect(soutputs[1]);
  iinputs[4].connect(newx.single());
}

void InterpExpo::interpolate(TransformationDescriptor& insegment, SingleOutput& x, SingleOutput& y, SingleOutput& newx){
  const auto& soutputs = static_cast<Handle&>(insegment).outputs();

  auto interp = this->t_["interp"];
  auto iinputs = interp.inputs();
  iinputs[0].connect(x.single());
  iinputs[1].connect(y.single());
  iinputs[2].connect(soutputs[0]);
  iinputs[3].connect(soutputs[1]);
  iinputs[4].connect(newx.single());
}

void InterpExpo::do_interpolate(FunctionArgs& fargs){
  auto& args=fargs.args;                                                  /// name inputs

  auto& points_a=args[4].x;                                               /// new x points
  auto  npoints=points_a.size();                                          /// number of points
  auto  point=points_a.data();                                            /// point read buffer
  auto& x_a=args[0].x;                                                    /// x of segments
  auto  x_buffer=x_a.data();                                              /// x's buffer
  auto& y_a=args[1].x;                                                    /// y of segments, exponent scale
  auto  y_buffer=y_a.data();                                              /// y's buffer
  auto& widths_a=args[3].x;                                               /// segment widths

  auto nseg=x_a.size()-1;                                                 /// number of segments
  auto b_a=((y_a.head(nseg)/y_a.tail(nseg)).log()/widths_a).eval();       /// b coefficient
  auto b_buffer=b_a.data();                                               /// b buffer

  auto insegment=args[2].buffer;                                          /// insegment buffer
  auto result=fargs.rets[0].buffer;                                       /// interpolation write buffer

  for(decltype(npoints) i{0}; i<npoints; ++i){
    auto idx = static_cast<size_t>(*insegment);
    if( *insegment<0 ){          /// underflow, extrapolate
      idx=0u;
    }
    else if( *insegment>=nseg ){ /// overflow, extrapolate
      idx=nseg-1u;
    }
    *result = *next(y_buffer, idx) * Exp((*next(x_buffer, idx) - *point)*(*next(b_buffer, idx)));

    advance(point, 1);
    advance(result, 1);
    advance(insegment, 1);
  }

}

//InterpExpo::Strategy InterpExpo::getStrategy(const std::string& strategy){
  //if(strategy=="constant"){
    //return Constant;
  //}
  //else if(strategy=="extrapolate"){
    //return Extrapolate;
  //}
  //else{
    //throw std::runtime_error("Unknown underflow/overflow strategy");
  //}

  //return Constant;
//}

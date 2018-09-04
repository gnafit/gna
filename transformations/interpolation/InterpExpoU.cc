#include "InterpExpoU.hh"
#include "TypesFunctions.hh"

#include <TMath.h>

//#define DEBUG_INTERPEXPOU

#include <stdexcept>

using std::next;
using std::prev;
using std::advance;
using TMath::Exp;

//InterpExpoU::InterpExpoU(const std::string& underflow_strategy, const std::string& overflow_strategy) : InSegment() {

InterpExpoU::InterpExpoU() : InSegment() {
  transformation_("interp")
    .input("newx")             /// 0
    .input("x")                /// 1
    .input("y")                /// 2
    .input("insegment")        /// 3
    .input("widths")           /// 4
    .output("interp")          /// 0
    .types(TypesFunctions::ifPoints<0>)                              /// newx is an array of any shape
    .types(TypesFunctions::ifPoints<1>, TypesFunctions::if1d<1>)     /// x is an 1d array
    .types(TypesFunctions::ifPoints<2>, TypesFunctions::if1d<2>)     /// y is an 1d array
    .types(TypesFunctions::ifPoints<3>, TypesFunctions::ifSameShape2<0,3>) /// segment index is of shape of newx
    .types(TypesFunctions::ifPoints<4>, TypesFunctions::if1d<4>)     /// widths is an 1d array
    .types(TypesFunctions::ifSame2<1,2>, TypesFunctions::ifBinsEdges<4,1>)
    .types(TypesFunctions::pass<0,0>)
    .func(&InterpExpoU::do_interpolate)
    ;

  //if(underflow_strategy.length()){
    //this->setUnderflowStrategy(underflow_strategy);
  //}
  //if(overflow_strategy.length()){
    //this->setOverflowStrategy(overflow_strategy);
  //}
}

void InterpExpoU::interpolate(SingleOutput& x, SingleOutput& y, SingleOutput& newx){
  auto segments = this->t_["segments"];
  auto sinputs  = segments.inputs();
  auto soutputs = segments.outputs();
  sinputs[0].connect(newx.single());
  sinputs[1].connect(x.single());

  auto interp = this->t_["interp"];
  auto iinputs = interp.inputs();
  iinputs[0].connect(newx.single());
  iinputs[1].connect(x.single());
  iinputs[2].connect(y.single());
  iinputs[3].connect(soutputs[0]);
  iinputs[4].connect(soutputs[1]);
}

void InterpExpoU::interpolate(TransformationDescriptor& segments, SingleOutput& x, SingleOutput& y, SingleOutput& newx){
  auto soutputs = static_cast<Handle&>(segments).outputs();

  auto interp = this->t_["interp"];
  auto iinputs = interp.inputs();
  iinputs[0].connect(newx.single());
  iinputs[1].connect(x.single());
  iinputs[2].connect(y.single());
  iinputs[3].connect(soutputs[0]);
  iinputs[4].connect(soutputs[1]);
}

void InterpExpoU::do_interpolate(FunctionArgs& fargs){
  auto& args=fargs.args;                                                  /// name inputs

  auto& points_a=args[0].x;                                               /// new x points
  auto  npoints=points_a.size();                                          /// number of points
  auto  point=points_a.data();                                            /// point read buffer
  auto& x_a=args[1].x;                                                    /// x of segments
  auto  x_buffer=x_a.data();                                              /// x's buffer
  auto& y_a=args[2].x;                                                    /// y of segments, exponent scale
  auto  y_buffer=y_a.data();                                              /// y's buffer
  auto& widths_a=args[4].x;                                               /// segment widths

  auto nseg=x_a.size()-1;                                                 /// number of segments
  auto b_a=((y_a.head(nseg)/y_a.tail(nseg)).log()/widths_a).eval();       /// b coefficient
  auto b_buffer=b_a.data();                                               /// b buffer

  auto insegment=args[3].buffer;                                          /// insegment buffer
  auto result=fargs.rets[0].buffer;                                       /// interpolation write buffer

  for(decltype(npoints) i{0}; i<npoints; ++i){
    auto idx = *insegment;
    if( *insegment<0 ){          /// underflow, extrapolate
      idx=0;
    }
    else if( *insegment>=nseg ){ /// overflow, extrapolate
      idx=nseg-1;
    }
    *result = *next(y_buffer, idx) * Exp((*next(x_buffer, idx) - *point)*(*next(b_buffer, idx)));

    advance(point, 1);
    advance(result, 1);
  }

  #ifdef DEBUG_INTERPEXPOU
  printf("Fill %i->%i (%i): fcn\n", (int)idx_current, (int)idx_next, (int)length);
  #endif
}

//InterpExpoU::Strategy InterpExpoU::getStrategy(const std::string& strategy){
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

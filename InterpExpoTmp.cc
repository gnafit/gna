#include "InterpExpo.hh"
#include "TypesFunctions.hh"
#include "fmt/format.h"

#include <TMath.h>
#include <stdexcept>

#ifdef GNA_CUDA_SUPPORT
#include "cuInterpExpo.hh"
#endif


using std::next;
using std::prev;
using std::advance;
using TMath::Exp;

//InterpExpo::InterpExpo(const std::string& underflow_strategy, const std::string& overflow_strategy) : InSegment() {

InterpExpo::InterpExpo() : InSegment() {
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

InterpExpo::InterpExpo(SingleOutput& x, SingleOutput& newx) : InterpExpo()
{
  set(x, newx);
  bind_inputs();
}

InterpExpo::InterpExpo(SingleOutput& x, SingleOutput& y, SingleOutput& newx) : InterpExpo()
{
  interpolate(x, y, newx);
}

TransformationDescriptor InterpExpo::add_transformation(const std::string& name){
  transformation_(new_transformation_name(name))
    .input("newx")             /// 0
    .input("x")                /// 1
    .input("insegment")        /// 2
    .input("widths")           /// 3
    .input("y")                /// 4
    .output("interp")          /// 0
    .types(TypesFunctions::ifPoints<0>)                                     /// newx is an array of any shape
    .types(TypesFunctions::ifPoints<1>, TypesFunctions::if1d<1>)            /// x is an 1d array
    .types(TypesFunctions::ifPoints<2>, TypesFunctions::ifSameShape2<0,2>)  /// segment index is of shape of newx
    .types(TypesFunctions::ifPoints<3>, TypesFunctions::if1d<3>)            /// widths is an 1d array
    .types(TypesFunctions::ifSame2<1,4>, TypesFunctions::ifBinsEdges<3,1>)
    .types(TypesFunctions::ifPoints<4>, TypesFunctions::if1d<4>)            /// y is an 1d array
    .types(TypesFunctions::ifSameInRange<4,-1,true>, TypesFunctions::passToRange<0,0,-1,true>)
    .func(&InterpExpo::do_interpolate)
#ifdef GNA_CUDA_SUPPORT 
    .func("gpu", &InterpExpo::do_interpolate_ongpu, DataLocation::Device)
#endif
    ;

  reset_open_input();
  bind_transformations();
  return transformations.back();
}

void InterpExpo::set(SingleOutput& x, SingleOutput& newx){
  auto segments = transformations.front();
  auto sinputs  = segments.inputs;
  sinputs[0].connect(newx.single());
  sinputs[1].connect(x.single());
}

void InterpExpo::bind_transformations(){
  auto segments=transformations.front();
  auto interp=transformations.back();

  auto& outputs=segments.outputs;
  auto& inputs=interp.inputs;

  outputs[0]>>inputs[2];
  outputs[1]>>inputs[3];
}

void InterpExpo::bind_inputs(){
  auto segments=transformations.front();
  auto interp=transformations.back();

  auto& seg_inputs=segments.inputs;
  auto& outputs=segments.outputs;
  auto& inputs=interp.inputs;

  seg_inputs[0].output()>>inputs[0];
  seg_inputs[1].output()>>inputs[1];
}

OutputDescriptor InterpExpo::interpolate(SingleOutput& x, SingleOutput& y, SingleOutput& newx){
  set(x, newx);
  auto output=add_input(y);
  bind_inputs();
  return output;
}

void InterpExpo::do_interpolate(FunctionArgs& fargs){
  auto& args=fargs.args;                                                  /// name inputs
  auto& rets=fargs.rets;                                                  /// name outputs

  auto& points_a=args[0].x;                                               /// new x points
  auto  npoints=points_a.size();                                          /// number of points
  auto& x_a=args[1].x;                                                    /// x of segments
  auto& widths_a=args[3].x;                                               /// segment widths

  auto nseg=x_a.size()-1;                                                 /// number of segments

  auto insegment_start=args[2].buffer;                                    /// insegment buffer

  for (size_t ret = 0; ret < rets.size(); ++ret) {
    auto insegment=insegment_start;                                       /// insegment buffer
    auto point=points_a.data();                                           /// point read buffer
    auto x_buffer=x_a.data();                                             /// x's buffer

    auto& y_a=args[4+ret].x;                                              /// y of segments, exponent scale
    auto  y_buffer=y_a.data();                                            /// y's buffer

    auto  b_a=((y_a.head(nseg)/y_a.tail(nseg)).log()/widths_a).eval();    /// b coefficient
    auto  b_buffer=b_a.data();                                            /// b buffer

    auto result=rets[ret].buffer;                                         /// interpolation write buffer

    for(decltype(npoints) i{0}; i<npoints; ++i){
      auto idx = static_cast<size_t>(*insegment);
      if( *insegment<0 ){          /// underflow, extrapolate
        idx=0u;
      }
      else if( *insegment>=nseg ){ /// overflow, extrapolate
        idx=nseg-1u;
      }
      *result = y_buffer[idx] * Exp((x_buffer[idx] - *point)*(b_buffer[idx]));

      advance(point, 1);
      advance(result, 1);
      advance(insegment, 1);
    }
  }
}

void InterpExpo::do_interpolate_ongpu(FunctionArgs& fargs) {
        auto& gpuargs=fargs.gpu;
	auto** source=gpuargs->args;
        auto** dest  =gpuargs->rets;
        auto** shape =gpuargs->argshapes;
        auto** rshape =gpuargs->retshapes;
        printf("%p %p %p %p\n", (void*)source, (void*)dest, (void*)shape, (void*)rshape);

/*	fargs.args.touch();
        gpuargs->provideSignatureDevice();
*/
	std::cout << "INTERP ON GPU" << std::endl;
	interpExpo_v1(gpuargs->args, gpuargs->rets, gpuargs->argshapes[0][0], gpuargs->argshapes[1][0] );
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
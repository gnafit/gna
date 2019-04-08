#include "InterpSpline.hh"

InterpSpline::InterpSpline() : InSegment() {
  add_transformation();
  add_input();
  set_open_input();
}

TransformationDescriptor InterpSpline::add_transformation(const std::string& name){
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
    .func(&InterpSpline::do_interpolate)
    ;

  reset_open_input();
  bind_transformations();
  return transformations.back();
}

void InterpSpline::set(SingleOutput& x, SingleOutput& newx){
  auto segments = transformations.front();
  auto sinputs  = segments.inputs;
  sinputs[0].connect(newx.single());
  sinputs[1].connect(x.single());
}

void InterpSpline::bind_transformations(){
  auto segments=transformations.front();
  auto interp=transformations.back();

  auto& outputs=segments.outputs;
  auto& inputs=interp.inputs;

  outputs[0]>>inputs[2];
  outputs[1]>>inputs[3];
}

void InterpSpline::bind_inputs(){
  auto segments=transformations.front();
  auto interp=transformations.back();

  auto& seg_inputs=segments.inputs;
  auto& inputs=interp.inputs;

  seg_inputs[0].output()>>inputs[0];
  seg_inputs[1].output()>>inputs[1];
}

void InterpSpline::do_interpolate(FunctionArgs& fargs) {
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

    auto& y_a=args[4+ret].x;                                              /// y of segments, the offset
    auto  y_buffer=y_a.data();                                            /// y's buffer

    auto  k_a=((y_a.tail(nseg)-y_a.head(nseg))/widths_a).eval();          /// k coefficient
    auto  k_buffer=k_a.data();                                            /// k buffer

    auto result=rets[ret].buffer;                                         /// interpolation write buffer


    for(decltype(npoints) i{0}; i<npoints; ++i){
      auto idx = static_cast<size_t>(*insegment);
      if( *insegment<0 ){          /// underflow, extrapolate
        idx=0u;
      }
      else if( *insegment>=nseg ){ /// overflow, extrapolate
        idx=nseg-1u;
      }
      *result = k_buffer[idx] * (*point - x_buffer[idx]) + y_buffer[idx];

      std::advance(point, 1);
      std::advance(result, 1);
      std::advance(insegment, 1);
    }
  }
}

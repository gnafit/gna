#include "InterpExpo.hh"
#include "TypesFunctions.hh"

#include <stdexcept>

using std::next;
using std::prev;

InterpExpo::InterpExpo(const std::string& underflow_strategy, const std::string& overflow_strategy) : SegmentWise() {
  transformation_("interp")
    .input("newx")
    .input("x")
    .input("y")
    .input("segments")
    .output("interp")
    .types(TypesFunctions::ifPoints<0>, TypesFunctions::if1d<0>)
    .types(TypesFunctions::ifPoints<1>, TypesFunctions::if1d<1>)
    .types(TypesFunctions::ifPoints<2>, TypesFunctions::if1d<2>)
    .types(TypesFunctions::ifPoints<3>, TypesFunctions::if1d<3>)
    .types(TypesFunctions::pass<0,0>)
    .func(&InterpExpo::do_interpolate)
    ;

  if(underflow_strategy.length()){
    this->setUnderflowStrategy(underflow_strategy);
  }
  if(overflow_strategy.length()){
    this->setOverflowStrategy(overflow_strategy);
  }
}

void InterpExpo::interpolate(SingleOutput& x, SingleOutput& y, SingleOutput& newx){
  auto segments = this->t_["segments"];
  segments.inputs["x"](x)
  segments.inputs["points"](newx)

  auto interp = this->t_["interp"];
  interp.inputs["points"](newx)
  interp.inputs["x"](x)
  interp.inputs["y"](y)
  interp.inputs["segments"](segments.outputs["segments"])
}

void InterpExpo::do_interpolate(Args args, Rets rets){
  auto& newx_a=args[0].x;
  auto& x_a=args[1].x;
  auto& y_a=args[2].x;

  auto nseg=x_a.size()-1;
  auto& ratio_a=y_a.head(nseg)/y_a.tail(nseg);
  auto& width_a=x_a.tail(nseg)-x_a.head(nseg);
  auto& b_a=(ratio_a.log()/width_a).eval();

  auto k_current=y_a.data();
  auto b_current=b_a.data();
  auto e0_current=x_a.data();

  auto& ddx_d=args[3];
  auto  ddx_current=ddx_d.buffer;
  auto  ddx_last=next(ddx_current, ddx_d.x.size()-1);
  auto  ddx_next=next(ddx_current);

  auto& interp_a=rets[0].x;

  auto idx_current=static_cast<size_t>(*ddx_current);
  if(idx_current>0u){
    interp_a.head(idx_current) = m_underflow;
    //printf("Fill %i->%i (%i): %g (head)\n", (int)0, (int)idx_current, (int)idx_current, m_underflow);
  }

  auto next_iter = [&](){
    ddx_current=next(ddx_current);
    ddx_next=next(ddx_next);
    idx_current=static_cast<size_t>(*ddx_current);

    k_current=next(k_current);
    b_current=next(b_current);
    e0_current=next(e0_current);
  };
  while(ddx_current<ddx_last){
    auto idx_next=static_cast<size_t>(*ddx_next);

    if(idx_current==idx_next){
      next_iter();
      continue;
    }

    auto length=idx_next-idx_current;

    //printf("Fill %i->%i (%i): fcn\n", (int)idx_current, (int)idx_next, (int)length);
    interp_a.segment(idx_current, length) = *k_current * ((*e0_current - newx_a.segment(idx_current, length))*(*b_current)).exp();

    next_iter();
  }

  auto n=static_cast<size_t>(interp_a.size());
  if(idx_current<n){
    interp_a.tail(n-idx_current)=m_overflow;
    //printf("Fill %i->%i (%i): %g (tail)\n", (int)idx_current, (int)n, (int)(n-idx_current), m_overflow);
  }
}

InterpExpo::Strategy InterpExpo::getStrategy(const std::string& strategy){
  if(strategy=="constant"){
    return Constant;
  }
  else if(strategy=="extrapolate"){
    return Extrapolate;
  }
  else{
    throw std::runtime_error("Unknown underflow/overflow strategy");
  }

  return Constant;
}

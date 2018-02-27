#include "InterpExpo.hh"
#include "TypesFunctions.hh"

//#define DEBUG_INTERPEXPO

#include <stdexcept>

using std::next;
using std::prev;

InterpExpo::InterpExpo(const std::string& underflow_strategy, const std::string& overflow_strategy) : SegmentWise() {
  transformation_("interp")
    .input("newx")             /// 0
    .input("x")                /// 1
    .input("y")                /// 2
    .input("segments")         /// 3
    .input("widths")           /// 4
    .output("interp")          ///
    .types(TypesFunctions::ifPoints<0>, TypesFunctions::if1d<0>)
    .types(TypesFunctions::ifPoints<1>, TypesFunctions::if1d<1>)
    .types(TypesFunctions::ifPoints<2>, TypesFunctions::if1d<2>)
    .types(TypesFunctions::ifPoints<3>, TypesFunctions::if1d<3>)
    .types(TypesFunctions::ifPoints<4>, TypesFunctions::if1d<4>)
    .types(TypesFunctions::ifSame2<1,2>, TypesFunctions::ifSame2<1,3>, TypesFunctions::ifBinsEdges<4,3>)
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

void InterpExpo::interpolate(TransformationDescriptor& segments, SingleOutput& x, SingleOutput& y, SingleOutput& newx){
  auto soutputs = segments.outputs();

  auto interp = this->t_["interp"];
  auto iinputs = interp.inputs();
  iinputs[0].connect(newx.single());
  iinputs[1].connect(x.single());
  iinputs[2].connect(y.single());
  iinputs[3].connect(soutputs[0]);
  iinputs[4].connect(soutputs[1]);
}

void InterpExpo::do_interpolate(Args args, Rets rets){
  auto& newx_a=args[0].x;
  auto& x_a=args[1].x;
  auto& y_a=args[2].x;
  auto& widths_a=args[4].x;

  auto nseg=x_a.size()-1;
  auto& b_a=((y_a.head(nseg)/y_a.tail(nseg)).log()/widths_a).eval();

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
    switch (m_underflow_strategy) {
      case Constant:
        interp_a.head(idx_current) = m_underflow;
        #ifdef DEBUG_INTERPEXPO
        printf("Fill %i->%i (%i): %g (head)\n", (int)0, (int)idx_current, (int)idx_current, m_underflow);
        #endif
        break;
      case Extrapolate:
        interp_a.head(idx_current) = *k_current * ((*e0_current - newx_a.head(idx_current))*(*b_current)).exp();
        #ifdef DEBUG_INTERPEXPO
        printf("Fill %i->%i (%i): fcn (head)\n", (int)0, (int)idx_current, (int)idx_current);
        #endif
        break;
    }
  }

  auto next_iter = [&](){
    ddx_current=next(ddx_current);
    ddx_next=next(ddx_next);
    idx_current=static_cast<size_t>(*ddx_current);

    if(ddx_current>=ddx_last){
      return false;
    }

    k_current=next(k_current);
    b_current=next(b_current);
    e0_current=next(e0_current);

    return true;
  };
  bool iterate=true;
  while(iterate){
    auto idx_next=static_cast<size_t>(*ddx_next);
    auto length=idx_next-idx_current;

    if(length==0u){
      iterate=next_iter();
      continue;
    }

    #ifdef DEBUG_INTERPEXPO
    printf("Fill %i->%i (%i): fcn\n", (int)idx_current, (int)idx_next, (int)length);
    #endif
    interp_a.segment(idx_current, length) = *k_current * ((*e0_current - newx_a.segment(idx_current, length))*(*b_current)).exp();

    iterate=next_iter();
  }

  auto nleft=interp_a.size()-static_cast<long int>(idx_current);
  if(nleft){
    switch (m_underflow_strategy) {
      case Constant:
        interp_a.tail(nleft)=m_overflow;
        #ifdef DEBUG_INTERPEXPO
        printf("Fill %i->%i (%i): %g (tail)\n", (int)idx_current, (int)interp_a.size(), (int)nleft, m_overflow);
        #endif
        break;
      case Extrapolate:
        interp_a.tail(nleft) = *k_current * ((*e0_current - newx_a.tail(nleft))*(*b_current)).exp();
        #ifdef DEBUG_INTERPEXPO
        printf("Fill %i->%i (%i): fcn (tail)\n", (int)idx_current, (int)interp_a.size(), (int)nleft);
        #endif
        break;
    }
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

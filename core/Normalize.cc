#include "Normalize.hh"
#include "TMath.h"
#include <Eigen/Core>
#include <cmath>

Normalize::Normalize() {
    transformation_(this, "normalize")
        .input("inp")
        .output("out")
        .types(Atypes::pass<0>)
        .func(&Normalize::doNormalize)
        ;
}

Normalize::Normalize(size_t start, size_t length) : m_start{start}, m_length{length} {
    transformation_(this, "normalize")
        .input("inp")
        .output("out")
        .types(Atypes::pass<0>, &Normalize::checkLimits)
        .func(&Normalize::doNormalize_segment)
        ;
}

void Normalize::doNormalize(Args args, Rets rets){
    auto& in=args[0].x;
    rets[0].x=in/in.sum();
}

void Normalize::doNormalize_segment(Args args, Rets rets){
    auto& in=args[0].x;
    rets[0].x=in/in.segment(m_start, m_length).sum();
}

void Normalize::checkLimits(Atypes args, Rtypes rets) {
    auto& dtype = args[0];
    if( dtype.shape.size()!=1u ){
        throw args.error(dtype, "Accept only 1d arrays in case a segment is specified");
    }
    auto length=dtype.shape[0];
    if( m_start>=length ){
        throw args.error(dtype, "Segment start is outside of the data limits");
    }
    if( (m_start+m_length)>length ){
        throw args.error(dtype, "Segment end is outside of the data limits");
    }
}

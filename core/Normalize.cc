#include "Normalize.hh"
#include "TMath.h"
#include <Eigen/Core>
#include <cmath>

Normalize::Normalize() {
    transformation_(this, "normalize")
        .input("input")
	.output("output")
	.types(Atypes::pass<0>)
	.func(&Normalize::normalize)
      ;
}

Normalize::Normalize(int start, int length) : m_start{start}, m_length{length} {
    transformation_(this, "normalize")
        .input("input")
	.output("output")
	.types(Atypes::pass<0>) //TODO: check limits and dimensions, add test
	.func(&Normalize::normalize_segment)
      ;
}

void Normalize::normalize(Args args, Rets rets){
    auto& res = rets[0].x = args[0].x;
    res/=res.sum();
}

void Normalize::normalize_segment(Args args, Rets rets){
    auto& res = rets[0].x = args[0].x;
    res/=res.segment(m_start, m_length).sum();
}

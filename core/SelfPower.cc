#include "SelfPower.hh"
#include "TMath.h"
#include <Eigen/Core>
#include <cmath>

SelfPower::SelfPower(const char* scalename/*="sp_scale"*/) {
    variable_(&m_scale, scalename);

    transformation_(this, "selfpower")
        .input("points")
	.output("result")
	.types(Atypes::ifPoints<0>,Atypes::pass<0>)
	.func(&SelfPower::calculate)
      ;

    transformation_(this, "selfpower_inv")
        .input("points")
	.output("result")
	.types(Atypes::ifPoints<0>,Atypes::pass<0>)
	.func(&SelfPower::calculate_inv)
      ;
}

void SelfPower::calculate(Args args, Rets rets){
    auto& res = rets[0].x = args[0].x/m_scale.value();
    res=res.pow(res);
}

void SelfPower::calculate_inv(Args args, Rets rets){
    auto& res = rets[0].x = args[0].x/m_scale.value();
    res=res.pow(-res);
}

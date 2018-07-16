#include <boost/math/constants/constants.hpp>
#include "HistSmear.hh"
#include "TypesFunctions.hh"

HistSmear::HistSmear(bool upper) {
  transformation_("smear")
      .input("Ntrue")
      .input("SmearMatrix")
      .output("Nvis")
      .types(TypesFunctions::if1d<0>, TypesFunctions::ifSquare<1>,  TypesFunctions::pass<0,0>)
      .types([](TypesFunctionArgs fargs) {
               auto& args=fargs.args;
               if (args[1].shape[0] != args[0].shape[0]) {
                 throw args.error(args[0], "SmearMatrix is not consistent with data vector");
               }
             })
       .func( upper ? &HistSmear::calcSmearUpper : &HistSmear::calcSmear );
}

void HistSmear::calcSmearUpper(FunctionArgs fargs) {
  auto& args=fargs.args;
  fargs.rets[0].x = args[1].mat.triangularView<Eigen::Upper>() * args[0].vec;
}

void HistSmear::calcSmear(FunctionArgs fargs) {
  auto& args=fargs.args;
  fargs.rets[0].x = args[1].mat * args[0].vec;
}

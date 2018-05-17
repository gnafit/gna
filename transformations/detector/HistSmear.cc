#include <boost/math/constants/constants.hpp>
#include "HistSmear.hh"
#include "TypesFunctions.hh"

HistSmear::HistSmear(bool upper) {
  transformation_("smear")
      .input("Ntrue")
      .input("SmearMatrix")
      .output("Nvis")
      .types(TypesFunctions::pass<0,0>,
         [](Atypes args, Rtypes /*rets*/) {
           if (args[1].shape.size() != 2) {
               throw args.error(args[0], "SmearMatrix is not matrix");
           }
           if (args[1].shape[0] != args[1].shape[1]) {
               throw args.error(args[0], "SmearMatrix is not square");
           }
           if (args[0].shape.size() != 1) {
               throw args.error(args[0], "Ntrue should be a vector");
           }
           if (args[1].shape[0] != args[0].shape[0]) {
               throw args.error(args[0], "SmearMatrix is not consistent with data vector");
           }
         })
       .func( upper ? &HistSmear::calcSmearUpper : &HistSmear::calcSmear );
}

void HistSmear::calcSmearUpper(Args args, Rets rets) {
  rets[0].x = args[1].mat.triangularView<Eigen::Upper>() * args[0].vec;
}

void HistSmear::calcSmear(Args args, Rets rets) {
  rets[0].x = args[1].mat * args[0].vec;
}

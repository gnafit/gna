#include "RenormalizeDiag.hh"

#include <iostream>
using namespace std;

RenormalizeDiag::RenormalizeDiag(size_t ndiag, bool upper, const char* parname) : m_ndiagonals(ndiag) {
  variable_(&m_diagscale, parname);

  transformation_(this, "renorm")
      .input("inmat")
      .output("outmat")
      .types(Atypes::pass<0>,
         [](Atypes args, Rtypes /*rets*/) {
           if (args[0].shape.size() != 2) {
               throw args.error(args[0], "SmearMatrix is not matrix");
           }
           if (args[0].shape[0] != args[0].shape[1]) {
               throw args.error(args[0], "SmearMatrix is not square");
           }
         })
     .func( upper ? &RenormalizeDiag::renormalizeUpper : &RenormalizeDiag::renormalize );
}

void RenormalizeDiag::renormalizeUpper(Args args, Rets rets) {
    rets[0].mat.triangularView<Eigen::StrictlyLower>().setZero();
    rets[0].mat.triangularView<Eigen::Upper>() = args[0].mat.triangularView<Eigen::Upper>();
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        rets[0].mat.diagonal(i)*=m_diagscale;
    }
    rets[0].arr2d.rowwise()/=rets[0].arr2d.colwise().sum();
}

void RenormalizeDiag::renormalize(Args args, Rets rets) {
    rets[0].arr = args[0].arr;
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        rets[0].mat.diagonal(i)*=m_diagscale;
        if( i>0 ) {
            rets[0].mat.diagonal(-i)*=m_diagscale;
        }
    }
    rets[0].arr2d.rowwise()/=rets[0].arr2d.colwise().sum();
}

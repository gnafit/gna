#include "RenormalizeDiag.hh"

RenormalizeDiag::RenormalizeDiag(size_t ndiag, bool triangular, const char* parname) : m_ndiagonals(ndiag) {
  variable_(&m_diagscale, parname);

  transformation_(this, "smear")
      .input("SmearMatrix")
      .output("Nvis")
      .types(Atypes::pass<0>,
         [](Atypes args, Rtypes /*rets*/) {
           if (args[0].shape.size() != 2) {
               throw args.error(args[0], "SmearMatrix is not matrix");
           }
           if (args[0].shape[0] != args[0].shape[1]) {
               throw args.error(args[0], "SmearMatrix is not square");
           }
         })
     .func( triangular ? &RenormalizeDiag::renormalizeTriangular : &RenormalizeDiag::renormalize );
}

void RenormalizeDiag::renormalizeTriangular(Args args, Rets rets) {
    rets[0].mat.triangularView<Eigen::StrictlyLower>().setZero();
    rets[0].mat.triangularView<Eigen::Upper>() = args[0].mat.triangularView<Eigen::Upper>();
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        rets[0].mat.diagonal(i)*=m_diagscale;
    }
    rets[0].arr.colwise()/=rets[0].arr.colwise().sum();
}

void RenormalizeDiag::renormalize(Args args, Rets rets) {
    rets[0].arr = args[0].arr;
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        rets[0].mat.diagonal(i)*=m_diagscale;
        if( i>0 ) {
            rets[0].mat.diagonal(-i)*=m_diagscale;
        }
    }
    rets[0].arr.colwise()/=rets[0].arr.colwise().sum();
}

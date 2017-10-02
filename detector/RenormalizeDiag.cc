#include "RenormalizeDiag.hh"

#include <iostream>
using namespace std;

RenormalizeDiag::RenormalizeDiag(size_t ndiag, Target target, Mode mode, const char* parname) : m_ndiagonals(ndiag) {
  variable_(&m_scale, parname);

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
     .func( mode==Mode::Upper ? ( target==Target::Offdiagonal ? &RenormalizeDiag::renormalizeOffdiagUpper
                                                              : &RenormalizeDiag::renormalizeDiagUpper )
                              : ( target==Target::Offdiagonal ? &RenormalizeDiag::renormalizeOffdiag
                                                              : &RenormalizeDiag::renormalizeDiag) );
}

void RenormalizeDiag::renormalizeOffdiagUpper(Args args, Rets rets) {
    if ( m_lower_uninitialized ){
        rets[0].mat.triangularView<Eigen::StrictlyLower>().setZero();
    }
    rets[0].mat.triangularView<Eigen::Upper>()=args[0].mat.triangularView<Eigen::Upper>();
    rets[0].mat.triangularView<Eigen::Upper>()*=m_scale;;
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        rets[0].mat.diagonal(i)=args[0].mat.diagonal(i);
    }
    rets[0].arr2d.rowwise()/=rets[0].arr2d.colwise().sum();
}

void RenormalizeDiag::renormalizeDiagUpper(Args args, Rets rets) {
    if ( m_lower_uninitialized ){
        rets[0].mat.triangularView<Eigen::StrictlyLower>().setZero();
    }
    rets[0].mat.triangularView<Eigen::Upper>() = args[0].mat.triangularView<Eigen::Upper>();
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        rets[0].mat.diagonal(i)*=m_scale;
    }
    rets[0].arr2d.rowwise()/=rets[0].arr2d.colwise().sum();
}

void RenormalizeDiag::renormalizeOffdiag(Args args, Rets rets) {
    rets[0].arr = args[0].arr*m_scale;
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        rets[0].mat.diagonal(i)=args[0].mat.diagonal(i);
        if( i>0 ) {
            rets[0].mat.diagonal(-i)=args[0].mat.diagonal(-i);
        }
    }
    rets[0].arr2d.rowwise()/=rets[0].arr2d.colwise().sum();
}

void RenormalizeDiag::renormalizeDiag(Args args, Rets rets) {
    rets[0].arr = args[0].arr;
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        rets[0].mat.diagonal(i)*=m_scale;
        if( i>0 ) {
            rets[0].mat.diagonal(-i)*=m_scale;
        }
    }
    rets[0].arr2d.rowwise()/=rets[0].arr2d.colwise().sum();
}

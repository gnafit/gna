#include "RenormalizeDiag.hh"

#include <iostream>
using namespace std;

RenormalizeDiag::RenormalizeDiag(size_t ndiag, bool triangular, const char* parname) : m_ndiagonals(ndiag) {
  variable_(&m_diagscale, parname);

  transformation_(this, "renorm")
      .input("inmat")
      .output("outmat")
      .types(Atypes::pass<0>,
         [](Atypes args, Rtypes /*rets*/) {
           if (args[0].shape.size() != 2) {
               throw args.error(args[0], "SmearMatrix is not matrix");
           }
           //if (args[0].shape[0] != args[0].shape[1]) {
               //throw args.error(args[0], "SmearMatrix is not square");
           //}
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
    cout<<"Source:"<<endl<<args[0].mat<<endl;
    cout<<"Target -1:"<<endl<<rets[0].mat<<endl;
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        rets[0].mat.diagonal(i)*=m_diagscale;
        if( i>0 ) {
            rets[0].mat.diagonal(-i)*=m_diagscale;
        }
        cout<<"  target it "<<i<<":"<<endl<<rets[0].mat<<endl;
    }
    cout<<"sum colwise:"<<endl<<rets[0].arr2d.colwise().sum()<<endl;
    rets[0].arr2d.rowwise()/=rets[0].arr2d.colwise().sum();
    cout<<"divide colwise:"<<endl<<rets[0].arr2d<<endl;
}

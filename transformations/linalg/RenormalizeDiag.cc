#include "RenormalizeDiag.hh"
#include <iostream>
#include <boost/format.hpp>
using namespace std;
using boost::format;

RenormalizeDiag::RenormalizeDiag(size_t ndiag, Target target, Mode mode, const char* parname) : m_ndiagonals(ndiag) {
  variable_(&m_scale, parname);

  auto memberFun = dispatchFunction(target, mode);
  if (memberFun == nullptr) {
      throw std::runtime_error((format("Can't dispatch the function in RenormalizeDiag! Passed target %1 and mode %2") % target % mode).str());
  }

  transformation_("renorm")
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
      .func(memberFun);
}

RenormalizeDiag::PointerToMember RenormalizeDiag::dispatchFunction(Target target, Mode mode) {
  RenormalizeDiag::PointerToMember dispatched = nullptr;
  switch (mode) {
      case Mode::Upper: {
          switch (target) {
              case Target::Offdiagonal:
                  dispatched = &RenormalizeDiag::renormalizeOffdiagUpper;
                  break;
              case Target::Diagonal:
                  dispatched = &RenormalizeDiag::renormalizeDiagUpper;
                  break;
          }
          break;
      };
      case Mode::Full: {
          switch (target) {
              case Target::Offdiagonal:
                  dispatched = &RenormalizeDiag::renormalizeOffdiag;
                  break;
              case Target::Diagonal:
                  dispatched = &RenormalizeDiag::renormalizeDiag;
                  break;
          }
          break;
      };
  };
  return dispatched;
}

double zero_to_one( double x ){
    return x==0.0 ? 1.0 : x;
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
    rets[0].arr2d.rowwise()/=rets[0].arr2d.colwise().sum().unaryExpr( std::ref(zero_to_one) ).eval();
}

void RenormalizeDiag::renormalizeDiagUpper(Args args, Rets rets) {
    if ( m_lower_uninitialized ){
        rets[0].mat.triangularView<Eigen::StrictlyLower>().setZero();
    }
    rets[0].mat.triangularView<Eigen::Upper>() = args[0].mat.triangularView<Eigen::Upper>();
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        rets[0].mat.diagonal(i)*=m_scale;
    }
    rets[0].arr2d.rowwise()/=rets[0].arr2d.colwise().sum().unaryExpr( std::ref(zero_to_one) ).eval();
}

void RenormalizeDiag::renormalizeOffdiag(Args args, Rets rets) {
    rets[0].arr = args[0].arr*m_scale;
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        rets[0].mat.diagonal(i)=args[0].mat.diagonal(i);
        if( i>0 ) {
            rets[0].mat.diagonal(-i)=args[0].mat.diagonal(-i);
        }
    }
    rets[0].arr2d.rowwise()/=rets[0].arr2d.colwise().sum().unaryExpr( std::ref(zero_to_one) ).eval();
}

void RenormalizeDiag::renormalizeDiag(Args args, Rets rets) {
    rets[0].arr = args[0].arr;
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        rets[0].mat.diagonal(i)*=m_scale;
        if( i>0 ) {
            rets[0].mat.diagonal(-i)*=m_scale;
        }
    }
    rets[0].arr2d.rowwise()/=rets[0].arr2d.colwise().sum().unaryExpr( std::ref(zero_to_one) ).eval();
}

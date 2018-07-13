#include "RenormalizeDiag.hh"
#include "TypesFunctions.hh"
#include <iostream>
#include <fmt/format.h>
using namespace std;

RenormalizeDiag::RenormalizeDiag(size_t ndiag, Target target, Mode mode, const char* parname) : m_ndiagonals(ndiag) {
  variable_(&m_scale, parname);

  auto memberFun = dispatchFunction(target, mode);
  if (memberFun == nullptr) {
      throw std::runtime_error((fmt::format("Can't dispatch the function in RenormalizeDiag! Passed target {0} and mode {1}", target, mode)));
  }

  transformation_("renorm")
      .input("inmat")
      .output("outmat")
      .types(TypesFunctions::pass<0>,
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

void RenormalizeDiag::renormalizeOffdiagUpper(FunctionArgs fargs) {
    auto& arg=fargs.args[0].mat;
    auto& ret=fargs.rets[0].mat;
    auto& retarr=fargs.rets[0].arr2d;
    if ( m_lower_uninitialized ){
        ret.triangularView<Eigen::StrictlyLower>().setZero();
    }
    ret.triangularView<Eigen::Upper>()=arg.triangularView<Eigen::Upper>();
    ret.triangularView<Eigen::Upper>()*=m_scale;;
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        ret.diagonal(i)=arg.diagonal(i);
    }
    retarr.rowwise()/=retarr.colwise().sum().unaryExpr( std::ref(zero_to_one) ).eval();
}

void RenormalizeDiag::renormalizeDiagUpper(FunctionArgs fargs) {
    auto& arg=fargs.args[0].mat;
    auto& ret=fargs.rets[0].mat;
    auto& retarr=fargs.rets[0].arr2d;
    if ( m_lower_uninitialized ){
        ret.triangularView<Eigen::StrictlyLower>().setZero();
    }
    ret.triangularView<Eigen::Upper>() = arg.triangularView<Eigen::Upper>();
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        ret.diagonal(i)*=m_scale;
    }
    retarr.rowwise()/=retarr.colwise().sum().unaryExpr( std::ref(zero_to_one) ).eval();
}

void RenormalizeDiag::renormalizeOffdiag(FunctionArgs fargs) {
    auto& arg=fargs.args[0].mat;
    auto& ret=fargs.rets[0].mat;
    auto& retarr=fargs.rets[0].arr2d;
    ret = arg*m_scale;
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        ret.diagonal(i)=arg.diagonal(i);
        if( i>0 ) {
            ret.diagonal(-i)=arg.diagonal(-i);
        }
    }
    retarr.rowwise()/=retarr.colwise().sum().unaryExpr( std::ref(zero_to_one) ).eval();
}

void RenormalizeDiag::renormalizeDiag(FunctionArgs fargs) {
    auto& arg=fargs.args[0].mat;
    auto& ret=fargs.rets[0].mat;
    auto& retarr=fargs.rets[0].arr2d;
    ret = arg;
    for (size_t i = 0; i < m_ndiagonals; ++i) {
        ret.diagonal(i)*=m_scale;
        if( i>0 ) {
            ret.diagonal(-i)*=m_scale;
        }
    }
    retarr.rowwise()/=retarr.colwise().sum().unaryExpr( std::ref(zero_to_one) ).eval();
}

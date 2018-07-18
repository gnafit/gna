#pragma once

#include "GNAObject.hh"

class RenormalizeDiag: public GNASingleObject,
                       public TransformationBind<RenormalizeDiag> {
public:
  enum Target {
    Diagonal = 0,
    Offdiagonal,
  };
  enum Mode {
    Full = 0,
    Upper,
  };

  RenormalizeDiag( size_t ndiag=1, Target target=Target::Diagonal, Mode mode=Mode::Upper, const char* parname="DiagScale" );

  using PointerToMember = void (RenormalizeDiag::*)(FunctionArgs&);

private:
  void renormalizeDiag(FunctionArgs& fargs);
  void renormalizeDiagUpper(FunctionArgs& fargs);
  void renormalizeOffdiag(FunctionArgs& fargs);
  void renormalizeOffdiagUpper(FunctionArgs& fargs);
  PointerToMember dispatchFunction(Target target, Mode mode);

  size_t m_ndiagonals;
  variable<double> m_scale;

  bool m_lower_uninitialized{true};
};

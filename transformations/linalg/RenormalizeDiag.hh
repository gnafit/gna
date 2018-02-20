#ifndef RENORMALIZEDIAG_H
#define RENORMALIZEDIAG_H

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

  using PointerToMember = void (RenormalizeDiag::*)(TransformationTypes::Args, TransformationTypes::Rets);

private:
  void renormalizeDiag(Args args, Rets rets);
  void renormalizeDiagUpper(Args args, Rets rets);
  void renormalizeOffdiag(Args args, Rets rets);
  void renormalizeOffdiagUpper(Args args, Rets rets);
  PointerToMember dispatchFunction(Target target, Mode mode);

  size_t m_ndiagonals;
  variable<double> m_scale;

  bool m_lower_uninitialized{true};
};

#endif // RENORMALIZEDIAG_H

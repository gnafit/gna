#ifndef RENORMALIZEDIAG_H
#define RENORMALIZEDIAG_H

#include "GNAObject.hh"

class RenormalizeDiag: public GNASingleObject,
                       public Transformation<RenormalizeDiag> {
public:
  RenormalizeDiag( size_t ndiag=1, bool upper=false, const char* parname="DiagScale" );

private:
  void renormalize(Args args, Rets rets);
  void renormalizeUpper(Args args, Rets rets);

  size_t m_ndiagonals;
  variable<double> m_diagscale;

  bool m_lower_uninitialized{true};
};

#endif // RENORMALIZEDIAG_H

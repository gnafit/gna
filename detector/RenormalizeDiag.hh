#ifndef RENORMALIZEDIAG_H
#define RENORMALIZEDIAG_H

#include "GNAObject.hh"

class RenormalizeDiag: public GNASingleObject,
                       public Transformation<RenormalizeDiag> {
public:
  RenormalizeDiag( size_t ndiag=1, bool utriangular=false, const char* parname="DiagScale" );

private:
  void renormalize(Args args, Rets rets);
  void renormalizeUtriangular(Args args, Rets rets);

  size_t m_ndiagonals;
  variable<double> m_diagscale;
};

#endif // RENORMALIZEDIAG_H

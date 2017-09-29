#ifndef RENORMALIZEDIAG_H
#define RENORMALIZEDIAG_H

#include "GNAObject.hh"

class RenormalizeDiag: public GNASingleObject,
                       public Transformation<RenormalizeDiag> {
public:
  RenormalizeDiag( size_t ndiag, bool triangular=false );

private:
  void renormalize(Args args, Rets rets);
  void renormalizeTriangular(Args args, Rets rets);

  size_t m_ndiagonals;
  variable<double> m_diagscale;
};

#endif // RENORMALIZEDIAG_H

#include "CovarianceToyMC.hh"
#include <boost/format.hpp>

CovarianceToyMC::CovarianceToyMC( bool autofreeze ) : m_autofreeze( autofreeze ) {
  transformation_(this, "toymc")
    .output("toymc")
    .types(&CovarianceToyMC::calcTypes)
    .func(&CovarianceToyMC::calcToyMC)
  ;
}

void CovarianceToyMC::add(SingleOutput &theory, SingleOutput &cov) {
  auto n = t_["toymc"].inputs().size()/2 + 1;
  t_["toymc"].input((boost::format("theory_%1%")%n).str()).connect(theory.single());
  t_["toymc"].input((boost::format("cov_%1%")%n).str()).connect(cov.single());
}

void CovarianceToyMC::nextSample() {
  t_["toymc"].unfreeze();
  t_["toymc"].taint();
}

void CovarianceToyMC::calcTypes(Atypes args, Rtypes rets) {
  if (args.size()%2 != 0) {
    throw args.undefined();
  }
  for (size_t i = 0; i < args.size(); i+=2) {
    if (args[i+0].shape.size() != 1) {
      throw rets.error(rets[0], "non-vector theory");
    }

    if (args[i+1].shape.size() != 2 ||
        args[i+1].shape[0] != args[i+1].shape[1]) {
      throw rets.error(rets[0], "incompatible covmat shape");
    }
    if (args[i+1].shape[0] != args[i+0].shape[0]) {
      throw rets.error(rets[0], "incompatible covmat shape 2");
    }
    rets[i/2] = args[i+0];
  }
}

void CovarianceToyMC::calcToyMC(Args args, Rets rets) {
  for (size_t i = 0; i < args.size(); i+=2) {
    auto &out = rets[i/2].vec;
    for (int j = 0; j < out.size(); ++j) {
      out(j) = m_gen( GNA::random::generator );
    }
    out = args[i+0].vec + args[i+1].mat.triangularView<Eigen::Lower>()*out;
  }
  if(m_autofreeze)
    rets.freeze();
}

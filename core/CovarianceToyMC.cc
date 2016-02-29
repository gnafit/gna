#include "CovarianceToyMC.hh"

CovarianceToyMC::CovarianceToyMC() {
  transformation_(this, "toymc")
    .output("toymc")
    .types(&CovarianceToyMC::calcTypes)
    .func(&CovarianceToyMC::calcToyMC)
  ;
}

void CovarianceToyMC::add(SingleOutput &theory, SingleOutput &cov) {
  t_["toymc"].input(theory);
  t_["toymc"].input(cov);
}

void CovarianceToyMC::nextSample() {
  t_["toymc"].unfreeze();
  t_["toymc"].taint();
}

void CovarianceToyMC::seed(unsigned int s) {
  m_rand.seed(s);
  m_gen.distribution().reset();
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
      out(j) = m_gen();
    }
    out = args[i+0].vec + args[i+1].mat.triangularView<Eigen::Lower>()*out;
  }
  rets.freeze();
}

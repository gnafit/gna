#include "CovarianceToyMC.hh"
#include <fmt/format.h>

CovarianceToyMC::CovarianceToyMC( bool autofreeze ) : m_autofreeze( autofreeze ) {
  transformation_("toymc")
    .output("toymc")
    .types(&CovarianceToyMC::calcTypes)
    .func(&CovarianceToyMC::calcToyMC)
  ;

  GNA::Random::register_callback( [this]{ this->m_distr.reset(); } );
}

void CovarianceToyMC::add(SingleOutput &theory, SingleOutput &cov) {
  auto n = t_["toymc"].inputs().size()/2 + 1;
  t_["toymc"].input(fmt::format("theory_{0}", n)).connect(theory.single());
  t_["toymc"].input(fmt::format("cov_{0}", n)).connect(cov.single());
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

void CovarianceToyMC::calcToyMC(FunctionArgs fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  for (size_t i = 0; i < args.size(); i+=2) {
    auto &out = rets[i/2].vec;
    for (int j = 0; j < out.size(); ++j) {
      out(j) = m_distr( GNA::Random::gen() );
    }
    out = args[i+0].vec + args[i+1].mat.triangularView<Eigen::Lower>()*out;
  }
  if(m_autofreeze)
    rets.freeze();
}

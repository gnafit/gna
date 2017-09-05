#include "PoissonToyMC.hh"

PoissonToyMC::PoissonToyMC( bool autofreeze ) : m_autofreeze( autofreeze ) {
  transformation_(this, "toymc")
    .output("toymc")
    .types(&PoissonToyMC::calcTypes)
    .func(&PoissonToyMC::calcToyMC)
  ;

  GNA::Random::register_callback( [=]{ this->m_distr.reset(); } );
}

void PoissonToyMC::add(SingleOutput &theory) {
  t_["toymc"].input(theory);
}

void PoissonToyMC::nextSample() {
  t_["toymc"].unfreeze();
  t_["toymc"].taint();
}

void PoissonToyMC::calcTypes(Atypes args, Rtypes rets) {
  for (size_t i = 0; i < args.size(); i+=1) {
    if (args[i].shape.size() != 1) {
      throw rets.error(rets[0], "non-vector theory");
    }
    rets[i] = args[i];
  }
}

void PoissonToyMC::calcToyMC(Args args, Rets rets) {
  for (size_t i = 0; i < args.size(); i+=1) {
    auto &mean = args[i].vec;
    auto &out = rets[i].vec;
    for (int j = 0; j < out.size(); ++j) {
      m_distr.param( std::poisson_distribution<>::param_type( mean(j) ) );
      out(j) = m_distr( GNA::Random::gen() );
    }
  }

  if(m_autofreeze)
    rets.freeze();
}

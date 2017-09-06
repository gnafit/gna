#include "NormalToyMC.hh"
#include <boost/format.hpp>

NormalToyMC::NormalToyMC( bool autofreeze ) : m_autofreeze( autofreeze ) {
  transformation_(this, "toymc")
    .output("toymc")
    .types(&NormalToyMC::calcTypes)
    .func(&NormalToyMC::calcToyMC)
  ;

  GNA::Random::register_callback( [=]{ this->m_distr.reset(); } );
}

void NormalToyMC::add(SingleOutput &theory, SingleOutput &sigma) {
  auto n = t_["toymc"].inputs().size()/2 + 1;
  t_["toymc"].input( (boost::format("theory_%1%")%n).str() ).connect(theory.single());
  t_["toymc"].input( (boost::format("sigma_%1%")%n).str() ).connect(sigma.single());
}

void NormalToyMC::nextSample() {
  t_["toymc"].unfreeze();
  t_["toymc"].taint();
}

void NormalToyMC::calcTypes(Atypes args, Rtypes rets) {
  if (args.size()%2 != 0) {
    throw args.undefined();
  }
  for (size_t i = 0; i < args.size(); i+=2) {
    if (args[i+0].shape.size() != 1) {
      throw rets.error(rets[0], "non-vector theory");
    }
    if (args[i+1].shape.size() != 1 ) {
      throw rets.error(rets[0], "incompatible sigma shape");
    }
    if (args[i+1].shape[0] != args[i+0].shape[0]) {
      throw rets.error(rets[0], "incompatible sigma shape 2");
    }
    rets[i/2] = args[i+0];
  }
}

void NormalToyMC::calcToyMC(Args args, Rets rets) {
  for (size_t i = 0; i < args.size(); i+=2) {
    auto &out = rets[i/2].arr;
    for (int j = 0; j < out.size(); ++j) {
      out(j) = m_distr( GNA::Random::gen() );
    }
    out = args[i+0].arr + args[i+1].arr*out;
  }

  if(m_autofreeze)
    rets.freeze();
}

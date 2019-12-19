#include "NormalStatsToyMC.hh"
#include "TypeClasses.hh"
#include <fmt/format.h>

NormalStatsToyMC::NormalStatsToyMC(bool autofreeze) :
GNAObjectBind1N("toymc", "theory", "toymc", 0, 0, 0),
  m_autofreeze(autofreeze) {
    this->add_transformation();
    this->add_input();
    this->set_open_input();

    GNA::Random::register_callback( [this]{ this->m_distr.reset(); } );
  }

void NormalStatsToyMC::nextSample() {
  for (size_t i = 0; i < this->transformations.size(); ++i) {
    auto trans = this->transformations[i];
    trans.unfreeze();
    trans.taint();
  }
}

void NormalStatsToyMC::calcToyMC(FunctionArgs fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  for (size_t i = 0; i < args.size(); ++i) {
    auto &in = args[i].arr;
    auto &out = rets[i].arr;
    for (int j = 0; j < out.size(); ++j) {
      out(j) = m_distr( GNA::Random::gen() );
    }
    out = in + in.sqrt()*out;
  }

  if(m_autofreeze) {
    rets.untaint();
    rets.freeze();
  }
}

TransformationDescriptor NormalStatsToyMC::add_transformation(const std::string& name){
  transformation_(new_transformation_name(name))
    .types(new TypeClasses::CheckNdimT<double>(1), new TypeClasses::PassEachTypeT<double>())
    .func(&NormalStatsToyMC::calcToyMC);

  reset_open_input();

  return transformations.back();
}

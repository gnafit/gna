#include "NormalToyMC.hh"
#include "TypeClasses.hh"
#include <fmt/format.h>

NormalToyMC::NormalToyMC(bool autofreeze) :
GNAObjectBindkN("toymc", {"theory", "sigma"}, "toymc", 0, 0, 0),
  m_autofreeze(autofreeze) {
    this->add_transformation();
    this->add_inputs();
    this->set_open_input();

    GNA::Random::register_callback( [this]{ this->m_distr.reset(); } );
  }

void NormalToyMC::nextSample() {
  for (size_t i = 0; i < this->transformations.size(); ++i) {
    auto trans = this->transformations[i];
    trans.unfreeze();
    trans.taint();
  }
}

void NormalToyMC::calcTypes(TypesFunctionArgs fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
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

void NormalToyMC::calcToyMC(FunctionArgs fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  for (size_t i = 0; i < args.size(); i+=2) {
    auto &out = rets[i/2].arr;
    for (int j = 0; j < out.size(); ++j) {
      out(j) = m_distr( GNA::Random::gen() );
    }
    out = args[i+0].arr + args[i+1].arr*out;
  }

  if(m_autofreeze) {
    rets.untaint();
    rets.freeze();
  }
}

TransformationDescriptor NormalToyMC::add_transformation(const std::string& name){
  transformation_(new_transformation_name(name))
    .types(new TypeClasses::PassEachTypeT<double>())
    .func(&NormalToyMC::calcToyMC);

  reset_open_input();

  return transformations.back();
}

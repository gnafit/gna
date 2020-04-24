#include "CovarianceToyMC.hh"
#include "TypeClasses.hh"
#include <fmt/format.h>

CovarianceToyMC::CovarianceToyMC( bool autofreeze ) :
GNAObjectBindkN("toymc", {"theory", "cov_L"}, "toymc", 0, 0, 0),
  m_autofreeze( autofreeze ) {
    this->add_transformation();
    this->add_inputs();
    this->set_open_input();

    GNA::Random::register_callback( [this]{ this->m_distr.reset(); } );
  }

void CovarianceToyMC::nextSample() {
  for (size_t i = 0; i < this->transformations.size(); ++i) {
    auto trans = this->transformations[i];
    trans.unfreeze();
    trans.taint();
  }
}

void CovarianceToyMC::calcTypes(TypesFunctionArgs fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
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
  if(m_autofreeze) {
    rets.untaint();
    rets.freeze();
  }
}

TransformationDescriptor CovarianceToyMC::add_transformation(const std::string& name){
  this->transformation_(new_transformation_name(name))
    .types(new TypeClasses::PassEachTypeT<double>({0,-1,2}))
    .types(&CovarianceToyMC::calcTypes)
    .func(&CovarianceToyMC::calcToyMC);

  reset_open_input();
  return transformations.back();
}

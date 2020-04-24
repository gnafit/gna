#include "PoissonToyMC.hh"
#include "TypeClasses.hh"
#include <fmt/format.h>

PoissonToyMC::PoissonToyMC(bool autofreeze) :
GNAObjectBind1N<double>("toymc", "theory", "toymc", 0, 0, 0),
m_autofreeze( autofreeze ) {
    this->add_transformation();
    this->add_input();
    this->set_open_input();

    GNA::Random::register_callback( [this]{ this->m_distr.reset(); } );
}

void PoissonToyMC::nextSample() {
    for (size_t i = 0; i < this->transformations.size(); ++i) {
        auto trans = this->transformations[i];
        trans.unfreeze();
        trans.taint();
    }
}

void PoissonToyMC::calcToyMC(FunctionArgs fargs) {
    auto& args=fargs.args;
    auto& rets=fargs.rets;
    for (size_t i = 0; i < args.size(); ++i) {
        auto &mean = args[i].vec;
        auto &out = rets[i].vec;
        for (int j = 0; j < out.size(); ++j) {
            m_distr.param(static_cast<typename decltype(m_distr)::param_type>(mean(j)));
            out(j) = m_distr(GNA::Random::gen());
        }
    }

    if(m_autofreeze) {
        rets.untaint();
        rets.freeze();
    }
}

TransformationDescriptor PoissonToyMC::add_transformation(const std::string& name){
    transformation_(new_transformation_name(name))
    .types(new TypeClasses::PassEachTypeT<double>())
    .func(&PoissonToyMC::calcToyMC);

    reset_open_input();
    return this->transformations.back();
}

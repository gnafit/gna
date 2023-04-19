#include "ParCenters.hh"
#include "fmt/ostream.h"

ParCenters::ParCenters(){
    transformation_("centers")
        .output("centers")
        .types([](ParCenters* obj, TypesFunctionArgs fargs){
               fargs.rets[0] = DataType().points().shape(obj->m_pars.size());})
        .func(&ParCenters::FillArray)
        .finalize();
}

ParCenters::ParCenters(std::vector<GaussianParameter<double>*> pars): ParCenters() {
    m_pars = pars;
    materialize();
}

void ParCenters::FillArray(FunctionArgs fargs) {
    auto& ret = fargs.rets[0].x;
    for (size_t i{0}; i < m_pars.size(); ++i) {
        auto* par = m_pars.at(i);
        ret(i) = par->central();
    }
}

void ParCenters::materialize() {
    t_["centers"].updateTypes();
}

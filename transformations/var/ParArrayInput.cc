#include "ParArrayInput.hh"
#include "fmt/ostream.h"
#include "TypeClasses.hh"

ParArrayInput::ParArrayInput() :
GNAObjectBindkN("pararray", {"points"}, "empty", 0, 0, -1)
{
    this->add_transformation();
    this->add_inputs();
    this->set_open_input();
}

TransformationDescriptor ParArrayInput::add_transformation(const std::string& name){
    using TypeClasses::CheckNdimT;
    this->transformation_(new_transformation_name(name))
        .types(new CheckNdimT<double>(1,0))
        .types(&ParArrayInput::CheckTypes)
        .func(&ParArrayInput::SetValues);

    reset_open_input();
    return transformations.back();
}

ParArrayInput::ParArrayInput(std::list<Parameter<double>*> pars): ParArrayInput() {
    m_pars_lists.emplace_back(pars);
}

InputDescriptor ParArrayInput::add_input(){
    if(!this->has_open_inputs()){
        m_pars_lists.push_back({});
    }
    GNAObjectBindkN::add_inputs();
    return InputDescriptor(transformations.back().inputs.back());
}

void ParArrayInput::CheckTypes(TypesFunctionArgs& fargs){
    auto& args = fargs.args;

    if (args.size()>m_pars_lists.size()){
        throw fargs.args.error(args[0],
            fmt::format("ParArrayInput: has {} inputs, but {} parameter sets",
                        args.size(), m_pars_lists.size()
                        )
            );
    }

    size_t i=0u;
    for (auto& pars: m_pars_lists) {
        if(i>=args.size()){
            break;
        }
        auto& arg = args[i]; ++i;

        if (arg.size()!=pars.size()){
            throw fargs.args.error(arg,
                fmt::format("ParArrayInput: input {} has {} parameters, but got {} values",
                            i, pars.size(), arg.size()
                            )
                );
        }
    }
}

void ParArrayInput::SetValues(FunctionArgs fargs) {
    auto& args = fargs.args;

    size_t i_list=0u;
    for (auto& pars: m_pars_lists) {
        if(i_list>=args.size()){
            break;
        }
        auto& arg = args[i_list].x; ++i_list;
        size_t i_par=0u;
        for (auto* par: pars) {
            par->set(arg(i_par));
            i_par++;
        }
    }
}

void ParArrayInput::materialize() {
    t_["pararray"].updateTypes();
}

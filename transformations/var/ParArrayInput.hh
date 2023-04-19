#pragma once
#include "GNAObject.hh"
#include "GNAObjectBindkN.hh"
#include <list>
#include "UncertainParameter.hh"

/**
 * @brief Provide an input array to set parameter values
 *
 * @author Maxim Gonchar
 * @date 2022.09
 */
class ParArrayInput: public GNAObjectBindkN,
                     public TransformationBind<ParArrayInput> {
public:
    ParArrayInput();
    ParArrayInput(std::list<Parameter<double>*> pars);

    TransformationDescriptor add_transformation(const std::string& name="");

    void append(Parameter<double>* par) {m_pars_lists.back().emplace_back(par);};
    void materialize();

    InputDescriptor add_input();

private:
    void CheckTypes(TypesFunctionArgs& args);
    std::list<std::list<Parameter<double>*>> m_pars_lists={{}};
    void SetValues(FunctionArgs fargs);
};


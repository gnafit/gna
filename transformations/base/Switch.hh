#pragma once

#include "GNAObject.hh"

/**
 * @brief Switch transformation passing only one of the inputs to the output
 *
 * TODO: dispatch taintflags. Only the current one should be unfrozen.
 *
 * @author Maxim Gonchar
 * @date 2021.03.12
 */
class Switch: public GNASingleObject,
              public TransformationBind<Switch> {
public:
    Switch(std::string varname);
    Switch(std::string varname, const typename OutputDescriptor::OutputDescriptors& outputs);

    InputDescriptor add_input(const char* name);
    InputDescriptor add_input(SingleOutput &data);
protected:
    void do_switch(FunctionArgs& fargs);

private:
    variable<double> m_choice;
};

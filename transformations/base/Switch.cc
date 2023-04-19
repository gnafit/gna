#include "Switch.hh"
#include "GNAObject.hh"
#include "TypeClasses.hh"
using namespace TypeClasses;

#include <algorithm>

Switch::Switch(std::string varname) {
  this->variable_(&m_choice, varname.data());

  this->transformation_("switch")
    .output("result")
    .types(new CheckSameTypesT<double>({0,-1}), new PassTypeT<double>(0, {0,0}))
    .func(&Switch::do_switch);
}

/**
 * @brief Construct Product from vector of SingleOutput instances
 */
Switch::Switch(std::string condition, const typename OutputDescriptor::OutputDescriptors& outputs) :
Switch(condition)
{
  for(auto output: outputs){
    this->add_input(output);
  }
}

/**
 * @brief The caclulation function
 */
void Switch::do_switch(FunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& ret=fargs.rets[0].x;

  int nelements=static_cast<int>(args.size());
  int choose=static_cast<int>(m_choice.value());
  if(choose<0 || choose>=nelements){
    throw fargs.rets.error("Swich: invalid index");
  }

  ret=args[choose].x;
}

InputDescriptor Switch::add_input(SingleOutput &out) {
  return InputDescriptor(transformations[0].input(out));
}

InputDescriptor Switch::add_input(const char* name) {
  return InputDescriptor(transformations[0].input(name));
}


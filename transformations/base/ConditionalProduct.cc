#include "ConditionalProduct.hh"
#include "GNAObject.hh"
#include "TypeClasses.hh"
using namespace TypeClasses;

#include <algorithm>

ConditionalProduct::ConditionalProduct(size_t nprod, std::string condition) : m_nprod(nprod) {
  if(nprod<1u){
    throw std::runtime_error("ConditionalProduct: number of elements should be at least 1");
  }

  this->variable_(&m_condition, condition.data());

  this->transformation_("product")
    .output("product")
    .types(new CheckSameTypesT<double>({0,-1}, "shape"), new PassTypeT<double>(0, {0,0}))
    .func(&ConditionalProduct::compute_product);
}

/**
 * @brief Construct Product from vector of SingleOutput instances
 */
ConditionalProduct::ConditionalProduct(size_t nprod, std::string condition, const typename OutputDescriptor::OutputDescriptors& outputs) :
ConditionalProduct(nprod, condition)
{
  for(auto output : outputs){
    this->multiply(output);
  }
}

/**
 * @brief The caclulation function
 */
void ConditionalProduct::compute_product(FunctionArgs& fargs) {
  auto& args=fargs.args;
  auto& ret=fargs.rets[0].x;
  ret = args[0].x;

  size_t nelements=args.size();
  if (m_condition.value()==0) {
    nelements = std::min(nelements, m_nprod);
  }

  for (size_t i = 1; i < nelements; ++i) {
    ret*=args[i].x;
  }
}

/**
 * @brief Add an input and connect it to the output.
 *
 * The input name is derived from the output name.
 *
 * @param out -- a SingleOutput instance.
 * @return InputDescriptor instance for the newly created input.
 */
InputDescriptor ConditionalProduct::multiply(SingleOutput &out) {
  return InputDescriptor(this->t_[0].input(out));
}

/**
 * @brief Add an input by name and leave unconnected.
 * @param name -- a name for the new input.
 * @return InputDescriptor instance for the newly created input.
 */
InputDescriptor ConditionalProduct::add_input(const char* name) {
  return InputDescriptor(this->t_[0].input(name));
}


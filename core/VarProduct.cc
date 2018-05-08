#include "VarProduct.hh"

/**
 * @brief Constructor.
 * @param varnames -- list of variable names.
 * @param productname -- variable name to store the result to.
 * @exception std::runtime_error in case less than 2 variables are passed.
 */
VarProduct::VarProduct(const std::vector<std::string>& varnames, const std::string& productname)
  : m_vars(varnames.size())
{
  if (m_vars.size()<2u) {
    throw std::runtime_error("You must specify at least two variables for VarProduct");
  }

  std::vector<changeable> deps;
  deps.reserve(varnames.size());
  for (size_t i = 0; i < varnames.size(); ++i) {
    variable_(&m_vars[i], varnames[i]);
    deps.push_back(m_vars[i]);
  }
  m_product = evaluable_<double>(productname, [this]() {
      double res = m_vars[0];
      for (size_t i = 1; i < m_vars.size(); ++i) {
          res*=m_vars[i];
      }
      return res;
    }, deps);
}

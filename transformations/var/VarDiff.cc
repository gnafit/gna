#include "VarDiff.hh"

/**
 * @brief Constructor.
 * @param varnames -- list of variable names.
 * @param diffname -- variable name to store the result to.
 * @param initial -- initial value for 'a'
 * @exception std::runtime_error in case less than 2 variables are passed.
 */
VarDiff::VarDiff(const std::vector<std::string>& varnames, const std::string& diffname, double initial)
  : VarDiff(varnames, diffname, true /*use initial*/) { m_initial=initial; }

/**
 * @brief Constructor.
 * @param varnames -- list of variable names.
 * @param diffname -- variable name to store the result to.
 * @exception std::runtime_error in case less than 2 variables are passed.
 */
VarDiff::VarDiff(const std::vector<std::string>& varnames, const std::string& diffname)
  : VarDiff(varnames, diffname, false /*ignore initial*/) {}

/**
 * @brief Constructor.
 * @param varnames -- list of variable names.
 * @param diffname -- variable name to store the result to.
 * @param useinitial -- if true: subtract all the variables from the initial value.
 * @exception std::runtime_error in case less than 2 variables are passed.
 */
VarDiff::VarDiff(const std::vector<std::string>& varnames, const std::string& diffname, bool useinitial)
  : m_vars(varnames.size())
{
  if (m_vars.size()<(2u-static_cast<size_t>(useinitial))) {
    throw std::runtime_error("You must specify at least two variables for VarDiff");
  }
  std::vector<changeable> deps;
  deps.reserve(varnames.size());
  for (size_t i = 0; i < varnames.size(); ++i) {
    variable_(&m_vars[i], varnames[i]);
    deps.push_back(m_vars[i]);
  }

  if (useinitial){
    m_diff = evaluable_<double>(diffname, [this]() {
        double res = m_vars[0];
        for (size_t i = 1; i < m_vars.size(); ++i) {
          res+=m_vars[i];
        }
        return m_initial-res;
      }, deps);
  }
  else
  {
    m_diff = evaluable_<double>(diffname, [this]() {
        double res = m_vars[0];
        for (size_t i = 1; i < m_vars.size(); ++i) {
          res-=m_vars[i];
        }
        return res;
      }, deps);
  }
}

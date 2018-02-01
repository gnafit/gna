#include "VarSum.hh"

VarSum::VarSum(const std::vector<std::string>& varnames, const std::string& sumname)
  : m_vars(varnames.size())
{
  std::vector<changeable> deps;
  deps.reserve(varnames.size());
  for (size_t i = 0; i < varnames.size(); ++i) {
    variable_(&m_vars[i], varnames[i]);
    deps.push_back(m_vars[i]);
  }
  m_sum = evaluable_<double>(sumname, [this]() {
      double res = 0.0;
      for (auto& var : m_vars) {
          res+=var;
      }
      return res;
    }, deps);
}

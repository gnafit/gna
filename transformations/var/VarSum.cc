#include "VarSum.hh"

/**
 * @brief Constructor.
 * @param varnames -- list of variable names.
 * @param sumname -- variable name to store the result to.
 * @exception std::runtime_error in case less than 2 variables are passed.
 */
template<typename FloatType>
GNA::GNAObjectTemplates::VarSumT<FloatType>::VarSumT(const std::vector<std::string>& varnames, const std::string& sumname)
: m_vars(varnames.size())
{
    if (m_vars.size()<2u) {
        throw std::runtime_error("You must specify at least two variables for VarSum");
    }

    std::vector<changeable> deps;
    deps.reserve(varnames.size());
    for (size_t i = 0; i < varnames.size(); ++i) {
        this->variable_(&m_vars[i], varnames[i]);
        deps.push_back(m_vars[i]);
    }
    m_sum = this->template evaluable_<FloatType>(sumname, [this]() {
                                      FloatType res = m_vars[0].value();
                                      for (size_t i = 1; i < m_vars.size(); ++i) {
                                          res+=m_vars[i].value();
                                      }
                                      return res;
                                  }, deps);
}

template class GNA::GNAObjectTemplates::VarSumT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::GNAObjectTemplates::VarSumT<float>;
#endif

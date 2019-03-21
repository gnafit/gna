#include "VarProduct.hh"

/**
 * @brief Constructor.
 * @param varnames -- list of variable names.
 * @param productname -- variable name to store the result to.
 * @exception std::runtime_error in case less than 2 variables are passed.
 */
    template<typename FloatType>
GNA::GNAObjectTemplates::VarProductT<FloatType>::VarProductT(const std::vector<std::string>& varnames, const std::string& productname)
    : m_vars(varnames.size())
{
    if (m_vars.size()<1u) {
        throw std::runtime_error("You must specify at least one variable for VarProduct");
    }

    std::vector<changeable> deps;
    deps.reserve(varnames.size());
    for (size_t i = 0; i < varnames.size(); ++i) {
        this->variable_(&m_vars[i], varnames[i]);
        deps.push_back(m_vars[i]);
    }
    m_product = this->template evaluable_<FloatType>(productname, [this]() {
                                                         FloatType res = m_vars[0].value();
                                                         for (size_t i = 1; i < m_vars.size(); ++i) {
                                                             res*=m_vars[i].value();
                                                         }
                                                         return res;
                                                     }, deps);
}

template class GNA::GNAObjectTemplates::VarProductT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
template class GNA::GNAObjectTemplates::VarProductT<float>;
#endif

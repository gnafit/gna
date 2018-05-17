#pragma once

#include "GNAObject.hh"

/**
 * @brief Transformation implementing the evaluable for variable sum 'a+b+c+...'.
 *
 * For at least two input variables computes the sum 'a+b+c+...'.
 *
 * @author Maxim Gonchar
 * @date 02.2018
 */
class VarSum: public GNAObject {
public:
  VarSum(const std::vector<std::string>& varnames, const std::string& sumname); ///< Constructor

protected:
  std::vector<variable<double>> m_vars; ///< List of variables to sum.
  dependant<double> m_sum;              ///< The sum result.
};

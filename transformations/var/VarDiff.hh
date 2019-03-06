#pragma once

#include "GNAObject.hh"

/**
 * @brief Transformation implementing the evaluable for variable difference 'a-b-c-...'.
 *
 * For at least two input variables computes the difference 'a-b-c-...'.
 *
 * @author Maxim Gonchar
 * @date 02.2018
 */
class VarDiff: public GNAObject {
public:
  VarDiff(const std::vector<std::string>& varnames, const std::string& diffname, double initial); ///< Constructor.
  VarDiff(const std::vector<std::string>& varnames, const std::string& diffname);                 ///< Constructor.

protected:
  VarDiff(const std::vector<std::string>& varnames, const std::string& diffname, bool use_initial); ///< Constructor.

  std::vector<variable<double>> m_vars; ///< List of variables.
  dependant<double> m_diff;             ///< The subtraction result.

  double m_initial;                     ///< Explicit initial value (for a).
};

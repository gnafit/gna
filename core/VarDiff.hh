#ifndef VARDIFF_H
#define VARDIFF_H

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
  VarDiff(const std::vector<std::string>& varnames, const std::string& diffname); ///< Constructor.

protected:
  std::vector<variable<double>> m_vars; ///< List of variables.
  dependant<double> m_diff;             ///< The subtraction result.
};

#endif // VARDIFF_H

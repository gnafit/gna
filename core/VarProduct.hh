#ifndef VARPRODUCT_H
#define VARPRODUCT_H

#include "GNAObject.hh"

/**
 * @brief Transformation implementing the evaluable for variable difference 'a*b*c*...'.
 *
 * For at least two input variables computes the product 'a*b*c*...'.
 *
 * @author Maxim Gonchar
 * @date 02.2018
 */
class VarProduct: public GNAObject {
public:
  VarProduct(const std::vector<std::string>& varnames, const std::string& productname); ///< Constructor.

protected:
  std::vector<variable<double>> m_vars; ///< List of variables.
  dependant<double> m_product;          ///< The product result.
};

#endif // VARPRODUCT_H

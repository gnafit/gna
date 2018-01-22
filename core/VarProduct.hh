#ifndef VARPRODUCT_H
#define VARPRODUCT_H

#include "GNAObject.hh"

class VarProduct: public GNAObject {
public:
  VarProduct(const std::vector<std::string>& varnamess, const std::string& productname);

protected:
  std::vector<variable<double>> m_vars;
  dependant<double> m_product;
};

#endif // VARPRODUCT_H

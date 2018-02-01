#ifndef VARSUM_H
#define VARSUM_H

#include "GNAObject.hh"

class VarSum: public GNAObject {
public:
  VarSum(const std::vector<std::string>& varnames, const std::string& sumname);

protected:
  std::vector<variable<double>> m_vars;
  dependant<double> m_sum;
};

#endif // VARSUM_H

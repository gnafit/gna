#ifndef VARDIFF_H
#define VARDIFF_H

#include "GNAObject.hh"

class VarDiff: public GNAObject {
public:
  VarDiff(const std::vector<std::string>& varnames, const std::string& diffname);

protected:
  std::vector<variable<double>> m_vars;
  dependant<double> m_diff;
};

#endif // VARDIFF_H

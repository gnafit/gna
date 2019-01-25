#pragma once

#include "GNAObject.hh"

class ReactorGroup: public GNAObject {
public:
  ReactorGroup(size_t count);
protected:
  std::vector<variable<double>> m_Ls;
  std::vector<variable<double>> m_Ps;

  dependant<double> m_Lavg;
  dependant<double> m_Pavg;
  dependant<double> m_weights;
};

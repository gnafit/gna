#pragma once

#include <vector>
#include <string>

#include "GNAObject.hh"

class ReactorNormAbsolute: public GNAObject,
                           public TransformationBind<ReactorNormAbsolute> {
public:
  ReactorNormAbsolute(const std::vector<std::string> &isonames);
protected:
  variable<double> m_norm;
};

class ReactorNorm: public GNAObject,
                   public TransformationBind<ReactorNorm> {
public:
  ReactorNorm(const std::vector<std::string> &isonames);
protected:
  void calcIsotopeNorms(FunctionArgs fargs);

  variable<double> m_thermalPower;
  std::vector<variable<double>> m_ePerFission;
  variable<double> m_targetProtons;
  variable<double> m_L;
};

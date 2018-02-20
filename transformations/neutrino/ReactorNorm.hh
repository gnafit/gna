#ifndef REACTORNORM_H
#define REACTORNORM_H

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
  void calcIsotopeNorms(Args args, Rets rets);

  variable<double> m_thermalPower;
  std::vector<variable<double>> m_ePerFission;
  variable<double> m_targetProtons;
  variable<double> m_L;
};

#endif // REACTORNORM_H

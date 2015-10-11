#ifndef OSCPROBPMNS_H
#define OSCPROBPMNS_H

#include <Eigen/Dense>

#include "GNAObject.hh"
#include "Neutrino.hh"

class OscillationVariables;
class PMNSVariables;
class OscProbPMNS: public GNAObject,
                   public Transformation<OscProbPMNS> {
public:
  OscProbPMNS(Neutrino from, Neutrino to);

  template <int I, int J>
  void calcComponent(Args args, Rets rets);
  void calcComponentCP(Args args, Rets rets);
  void calcSum(Args args, Rets rets);
protected:
  template <int I, int J>
  double DeltaMSq() const;
  template <int I, int J>
  double weight() const;
  double weightCP() const;

  variable<double> m_L;
  std::unique_ptr<OscillationVariables> m_param;
  std::unique_ptr<PMNSVariables> m_pmns;

  int m_alpha, m_beta;
};

#endif // OSCPROBPMNS_H

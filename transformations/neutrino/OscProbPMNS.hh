#pragma once

#include <Eigen/Dense>

#include "GNAObject.hh"
#include "Neutrino.hh"

class OscillationVariables;
class PMNSVariables;
class OscProbPMNSBase: public GNAObject,
                       public TransformationBind<OscProbPMNSBase> {
protected:
  OscProbPMNSBase(Neutrino from, Neutrino to);

  template <int I, int J>
  double DeltaMSq() const;

  template <int I, int J>
  double weight() const;

  double weightCP() const;

  std::unique_ptr<OscillationVariables> m_param;
  std::unique_ptr<PMNSVariables> m_pmns;

  int m_alpha, m_beta;
};

class OscProbPMNS: public OscProbPMNSBase,
                   public TransformationBind<OscProbPMNS> {
public:
  using TransformationBind<OscProbPMNS>::transformation_;

  OscProbPMNS(Neutrino from, Neutrino to, std::string l_name="L");

  template <int I, int J>
  void calcComponent(Args args, Rets rets);
  void calcComponentCP(Args args, Rets rets);
  void calcSum(Args args, Rets rets);
  void calcFullProb(Args args, Rets rets);
protected:
  variable<double> m_L;
};

class OscProbAveraged: public OscProbPMNSBase,
                       public TransformationBind<OscProbAveraged> {
public:
  using TransformationBind<OscProbAveraged>::transformation_;

  OscProbAveraged(Neutrino from, Neutrino to);
private:
  void CalcAverage(Args args, Rets rets);
};

class OscProbPMNSMult: public OscProbPMNSBase,
                       public TransformationBind<OscProbPMNSMult> {
public:
  using TransformationBind<OscProbPMNSMult>::transformation_;

  OscProbPMNSMult(Neutrino from, Neutrino to, std::string l_name="Lavg");

  template <int I, int J>
  void calcComponent(Args args, Rets rets);
  void calcSum(Args args, Rets rets);
protected:
  variable<double> m_Lavg;

  variable<std::array<double, 3>> m_weights;
};

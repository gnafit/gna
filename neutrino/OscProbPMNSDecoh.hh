#ifndef OSCPROBPMNSDECOH_H
#define OSCPROBPMNSDECOH_H

#include <Eigen/Dense>

#include "OscProbPMNS.hh"
#include "Neutrino.hh"

class OscillationVariables;
class PMNSVariables;
class OscProbPMNSDecoh: public OscProbPMNSBase,
                        public Transformation<OscProbPMNSDecoh> {
public:	
  OscProbPMNSDecoh(Neutrino from, Neutrino to);
  void calcSum(Args args, Rets rets);
  template <int I, int J>
  void calcComponent(Args args, Rets rets);
  template <int I, int J>
  void calcComponentCP(Args args, Rets rets);
protected:
  variable<double> m_L;
  variable<double> m_sigma; 		
};

#endif // OSCPROBPMNSDECOH_H


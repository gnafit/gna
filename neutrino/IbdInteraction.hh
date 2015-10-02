#ifndef IBDINTERACTION_H
#define IBDINTERACTION_H

#include <stdexcept>
#include <utility>

#include "GNAObject.hh"

class PDGVariables;
class IbdInteraction: public GNAObject {
public:
  IbdInteraction();
protected:
  PDGVariables *m_pdg;
  dependant<double> m_threshold;
  dependant<double> m_DeltaNP;
  dependant<double> m_NucleonMass;

  double m_cmfactor;

  const double PhaseFactor = 1.7152;

  const double g = 1.2601; // g_A/g_V: axial-vector and vector coupling constants ratio
  const double f = 1.;     // g_V/g_V
  const double f2 = 3.706; // \mu_p - \mu_n: isovector anomalous magnetic moment

  const double gsq = g*g;
  const double fsq = f*f;
  const double f2sq = f2*f2;

  ClassDef(IbdInteraction, 1);
};

#endif // IBDINTERACTION_H

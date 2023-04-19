#pragma once

#include "IbdInteraction.hh"

class IbdFirstOrder: public IbdInteraction,
                     public TransformationBind<IbdFirstOrder> {
public:
  IbdFirstOrder() { init(); }
  IbdFirstOrder(double PhaseFactor, double g, double f, double f2) :
    IbdInteraction(PhaseFactor, g, f, f2) { init(); }

protected:
  void calc_Enu(FunctionArgs fargs);
  double Xsec(double Eneu, double ctheta);
  void calc_Xsec(FunctionArgs fargs);
  void calc_dEnu_wrt_Ee(FunctionArgs fargs);

  dependant<double> ElectronMass2, NeutronMass2, ProtonMass2;
  dependant<double> DeltaNPE_tilded;
  dependant<double> y2;
  dependant<double> Enu_threshold;

private:
  void init();
};

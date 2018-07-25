#pragma once

#include "IbdInteraction.hh"

class IbdFirstOrder: public IbdInteraction,
                     public TransformationBind<IbdFirstOrder> {
public:
  IbdFirstOrder();
protected:
  void calc_Enu(FunctionArgs fargs);
  double Xsec(double Eneu, double ctheta);
  void calc_Xsec(FunctionArgs fargs);
  void calc_dEnu_wrt_Ee(FunctionArgs fargs);

  dependant<double> ElectronMass2, NeutronMass2, ProtonMass2;
  dependant<double> DeltaNPE_tilded;
  dependant<double> y2;
};

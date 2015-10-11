#ifndef IBDFIRSTORDER_H
#define IBDFIRSTORDER_H

#include "IbdInteraction.hh"

class IbdFirstOrder: public IbdInteraction,
                     public Transformation<IbdFirstOrder> {
public:
  IbdFirstOrder();
protected:
  void calc_Enu(Args args, Rets rets);
  double Xsec(double Eneu, double ctheta);
  void calc_Xsec(Args args, Rets rets);
  void calc_dEnu_wrt_Ee(Args args, Rets rets);

  dependant<double> ElectronMass2, NeutronMass2, ProtonMass2;
  dependant<double> DeltaNPE_tilded;
  dependant<double> y2;
};

#endif // IBDFIRSTORDER_H

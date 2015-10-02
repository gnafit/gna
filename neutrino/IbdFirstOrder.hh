#ifndef IBDFIRSTORDER_H
#define IBDFIRSTORDER_H

#include "IbdInteraction.hh"

class IbdFirstOrder: public IbdInteraction,
                     public Transformation<IbdFirstOrder> {
public:
  IbdFirstOrder();
protected:
  Status calc_Enu(Args args, Rets rets);
  double Xsec(double Eneu, double ctheta);
  Status calc_Xsec(Args args, Rets rets);
  Status calc_dEnu_wrt_Ee(Args args, Rets rets);

  dependant<double> ElectronMass2, NeutronMass2, ProtonMass2;
  dependant<double> DeltaNPE_tilded;
  dependant<double> NeutronLifeTimeMeV;
  dependant<double> y2;

  ClassDef(IbdFirstOrder, 1);
};

#endif // IBDFIRSTORDER_H

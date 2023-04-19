#pragma once

#include "IbdInteraction.hh"

class IbdZeroOrder: public IbdInteraction,
                    public TransformationBind<IbdZeroOrder> {
public:
  IbdZeroOrder() { init(); }
  IbdZeroOrder(bool a_useEnu): useEnu(a_useEnu) { init(); }
  IbdZeroOrder(double PhaseFactor, double g, double f, double f2) :
    IbdInteraction(PhaseFactor, g, f, f2) { init(); }
protected:
  void calcEnu(FunctionArgs fargs);
  void calcXsec(FunctionArgs fargs);
private:
  bool useEnu = false;

  void init();
};

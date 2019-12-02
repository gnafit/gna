#pragma once

#include "IbdInteraction.hh"

class IbdZeroOrder: public IbdInteraction,
                    public TransformationBind<IbdZeroOrder> {
public:
  IbdZeroOrder();
  IbdZeroOrder(bool useEnu);
protected:
  void calcEnu(FunctionArgs fargs);
  void calcXsec(FunctionArgs fargs);
private:
  bool useEnu = false;
};

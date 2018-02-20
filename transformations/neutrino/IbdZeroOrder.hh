#pragma once

#include "IbdInteraction.hh"

class IbdZeroOrder: public IbdInteraction,
                    public TransformationBind<IbdZeroOrder> {
public:
  IbdZeroOrder();
protected:
  void calcEnu(Args args, Rets rets);
  void calcXsec(Args args, Rets rets);
};

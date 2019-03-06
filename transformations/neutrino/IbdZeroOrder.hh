#pragma once

#include "IbdInteraction.hh"

class IbdZeroOrder: public IbdInteraction,
                    public TransformationBind<IbdZeroOrder> {
public:
  IbdZeroOrder();
protected:
  void calcEnu(FunctionArgs fargs);
  void calcXsec(FunctionArgs fargs);
};

#ifndef IBDZEROORDER_H
#define IBDZEROORDER_H

#include "IbdInteraction.hh"

class IbdZeroOrder: public IbdInteraction,
                    public Transformation<IbdZeroOrder> {
public:
  IbdZeroOrder();
protected:
  void calcEnu(Args args, Rets rets);
  void calcXsec(Args args, Rets rets);
private:
  ClassDef(IbdZeroOrder, 1);
};

#endif // IBDZEROORDER_H

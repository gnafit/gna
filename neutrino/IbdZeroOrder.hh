#ifndef IBDZEROORDER_H
#define IBDZEROORDER_H

#include "IbdInteraction.hh"

class IbdZeroOrder: public IbdInteraction,
                    public Transformation<IbdZeroOrder> {
public:
  IbdZeroOrder();
protected:
  Status calcEnu(Args args, Rets rets);
  Status calcXsec(Args args, Rets rets);
private:
  ClassDef(IbdZeroOrder, 1);
};

#endif // IBDZEROORDER_H

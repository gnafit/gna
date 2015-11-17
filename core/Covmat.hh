#ifndef COVMAT_H
#define COVMAT_H

#include "GNAObject.hh"

class Covmat: public GNAObject,
              public Transformation<Covmat> {
public:
  Covmat()
    : m_fixed(false)
  {
    transformation_(this, "cov")
      .types(Atypes::ifSame, [](Atypes args, Rtypes rets) {
          rets[0] = DataType().points().shape(args[0].size(), args[0].size());
        })
      .func(&Covmat::calculateCov)
      .input("stat")
      .output("cov")
    ;
    transformation_(this, "inv")
      .types(Atypes::pass<0>)
      .func(&Covmat::calculateInv)
      .input("cov")
      .output("inv")
      ;
  }
  void calculateCov(Args args, Rets rets);
  void calculateInv(Args args, Rets rets);

  void rank1(SingleOutput &data);
  // void rank1(SingleOutput &out);

  size_t ndim() const;
  size_t size() const;

  void update() const;
  const double *data() const;

  void setFixed(bool fixed) { m_fixed = fixed; }
protected:
  bool m_fixed;
};

#endif // COVMAT_H

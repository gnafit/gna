#ifndef COVMAT_H
#define COVMAT_H

#include "GNAObject.hh"

class Covmat: public GNAObject,
              public Transformation<Covmat> {
public:
  Covmat() {
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

  void rank1(GNASingleObject &obj) {
    rank1(obj[0].outputs.single());
  }
  void rank1(const TransformationDescriptor &obj) {
    rank1(obj.outputs.single());
  }
  void rank1(const TransformationDescriptor::Outputs &outs) {
    rank1(outs.single());
  }
  void rank1(const OutputDescriptor &single);

  size_t ndim() const;
  size_t size() const;

  void update() const;
  const double *data() const;
};

#endif // COVMAT_H

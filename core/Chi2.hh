#ifndef CHI2_H
#define CHI2_H

#include "GNAObject.hh"
#include "Statistic.hh"
#include <chrono>

class Chi2: public GNASingleObject,
            public Transformation<Chi2>,
            public Statistic {
public:
  Chi2() : InvMatCompFlag(false) {
    transformation_(this, "chi2")
      .output("chi2")
      .types(&Chi2::checkTypes)
      .func(&Chi2::calculateChi2)
    ;
    m_transform = t_["chi2"];
    t0 = Time::now();
  }

  void add(SingleOutput &theory, SingleOutput &data, SingleOutput &cov);

  void checkTypes(Atypes args, Rtypes rets);
  void calculateChi2(Args args, Rets rets);

  virtual double value() override {
    return m_transform[0].x[0];
  }
protected:
  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::milliseconds ms;
  typedef std::chrono::duration<float> fsec;
  std::chrono::time_point<std::chrono::system_clock> t0, t1;
  bool InvMatCompFlag;
  std::vector<Eigen::MatrixXd> L;
  Handle m_transform;
};

#endif // CHI2_H

#include "Covmat.hh"

void Covmat::calculateCov(Args args, Rets rets) {
  auto cov = rets[0].as2d();
  cov.setZero();
  cov.matrix().diagonal() = args[0].x.matrix();
  for (size_t i = 1; i < args.size(); ++i) {
    // cov.matrix().selfadjointView<Eigen::Upper>().rankUpdate(args[i].x.matrix());
    cov.matrix() += args[i].x.matrix()*args[i].x.matrix().transpose();
  }
  if (m_fixed) {
    rets.freeze();
  }
}

void Covmat::calculateInv(Args args, Rets rets) {
  rets[0].as2d().matrix() = args[0].as2d().matrix().inverse();
}

void Covmat::rank1(const OutputDescriptor &out) {
  t_["cov"].input(out.channel()).connect(out);
}

size_t Covmat::ndim() const {
  return t_["cov"][0].type.shape[0];
}

size_t Covmat::size() const {
  return t_["cov"][0].type.size();
}

void Covmat::update() const {
  t_["cov"].update(0);
}

const double *Covmat::data() const {
  return t_["cov"][0].x.data();
}

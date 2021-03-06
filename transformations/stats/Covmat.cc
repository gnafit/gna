#include "Covmat.hh"

void Covmat::calculateCov(FunctionArgs fargs) {
  auto& args=fargs.args;
  auto& rets=fargs.rets;
  auto &cov = rets[0].mat;
  cov.setZero();
  cov.diagonal() = args[0].vec;
  for (size_t i = 1; i < args.size(); ++i) {
    // cov.matrix().selfadjointView<Eigen::Upper>().rankUpdate(args[i].x.matrix());
    cov += args[i].vec*args[i].vec.transpose();
  }
  if (m_fixed) {
    rets.untaint();
    rets.freeze();
  }
}

void Covmat::calculateInv(FunctionArgs fargs) {
  fargs.rets[0].mat = fargs.args[0].mat.inverse();
}

void Covmat::prepareCholesky(TypesFunctionArgs fargs) {
  m_llt = LLT(fargs.args[0].shape[0]);
  fargs.rets[0].preallocated(const_cast<double*>(m_llt.matrixRef().data()));
}

void Covmat::calculateCholesky(FunctionArgs fargs) {
  m_llt.compute(fargs.args[0].mat);
}

void Covmat::rank1(SingleOutput &out) {
  t_["cov"].input(out);
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

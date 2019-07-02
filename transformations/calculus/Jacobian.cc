#include "Jacobian.hh"
#include "EigenHelpers.hh"
#include <algorithm>

void Jacobian::calcJacobian(FunctionArgs fargs) {
    auto& arg = fargs.args;
    auto& to_ret = fargs.rets;

    Eigen::MatrixXd storage(arg[0].x.size(), m_pars.size());
    storage.setZero();
    for (size_t i=0; i < m_pars.size(); ++i) {
      auto* x = m_pars.at(i);
      auto x0 = x->value();

      double f1 = 4.0/(3.0*m_reldelta);
      double f2 = 1.0/(6.0*m_reldelta);

      std::array<double, 4> points;
      points[0] = x->relativeValue(+m_reldelta/2);
      points[1] = x->relativeValue(-m_reldelta/2);
      points[2] = x->relativeValue(+m_reldelta);
      points[3] = x->relativeValue(-m_reldelta);

      x->set(points[0]);

      Eigen::ArrayXd ret = f1*arg[0].x;

      x->set(points[1]);
      ret -= f1*arg[0].x;
      x->set(points[2]);
      ret -= f2*arg[0].x;
      x->set(points[3]);
      ret += f2*arg[0].x;
      x->set(x0);

      storage.col(i) = ret.matrix();
    }

    to_ret[0].mat = storage;


}

void Jacobian::calcTypes(TypesFunctionArgs fargs){
    fargs.rets[0] = DataType().points().shape(fargs.args[0].points().shape()[0], m_pars.size());
}

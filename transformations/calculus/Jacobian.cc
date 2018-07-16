#include "Jacobian.hh"
#include "EigenHelpers.hh"
#include <algorithm>

void Jacobian::calcJacobian(FunctionArgs fargs) {
    auto& arg=fargs.args[0];
    auto& ret=fargs.rets[0];

    Eigen::MatrixXd storage(arg.x.size(), m_pars.size());
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

      Eigen::ArrayXd ret = f1*arg.x;

      x->set(points[1]);
      ret -= f1*arg.x;
      x->set(points[2]);
      ret -= f2*arg.x;
      x->set(points[3]);
      ret += f2*arg.x;
      x->set(x0);

      storage.col(i) = ret.matrix();
    }

    ret.mat = storage;
}

void Jacobian::calcTypes(TypesFunctionArgs fargs){
    fargs.rets[0] = DataType().points().shape(fargs.args[0].size(), m_pars.size());
}

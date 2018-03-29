#include "Jacobian.hh"
#include "EigenHelpers.hh"

void Jacobian::calcJacobian(Args args, Rets rets) {
    /* std::cout << "Dimensions of jac -- " << rets[0].mat.rows() << " x " << rets[0].mat.cols() << std::endl; */
    /* std::cout << "Initial jac " << rets[0].mat << std::endl; */
    Eigen::MatrixXd storage(args[0].x.size(), m_pars.size());
    storage.setZero();
    for (size_t i=0; i < m_pars.size(); ++i) {
      auto* x = m_pars.at(i);
      auto x0 = x->value();
      /* std::cout << "x0 = " << x0 << std::endl; */

      double f1 = 4.0/(3.0*m_reldelta);
      double f2 = 1.0/(6.0*m_reldelta);

      std::array<double, 4> points;
      points[0] = x->relativeValue(+m_reldelta/2);
      points[1] = x->relativeValue(-m_reldelta/2);
      points[2] = x->relativeValue(+m_reldelta);
      points[3] = x->relativeValue(-m_reldelta);

      x->set(points[0]);

      Eigen::ArrayXd ret = f1*args[0].x;

      x->set(points[1]);
      ret -= f1*args[0].x;
      x->set(points[2]);
      ret -= f2*args[0].x;
      x->set(points[3]);
      ret += f2*args[0].x;
      x->set(x0);

      storage.col(i) = ret.matrix();
    }

    rets[0].mat = storage;
}

void Jacobian::calcTypes(Atypes args, Rtypes rets){
    rets[0] = DataType().points().shape(args[0].size(), m_pars.size());
}

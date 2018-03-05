#include "Jacobian.hh"

void Jacobian::calcJacobian(Args args, Rets rets) {
    std::cout << "Dimensions of jac -- " << rets[0].mat.rows() << " x " << rets[0].mat.cols() << std::endl;
    std::cout << "Initial jac " << rets[0].mat << std::endl;
    const auto& input = args[0];
    auto size = input.x.size();
    std::cout << "Size input " << size << std::endl;
    for (size_t i=0; i < m_pars.size(); ++i) {
       /* rets[0].mat.col(i) = computeDerivative(args[0], m_pars.at(i)).matrix(); */
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

      Eigen::ArrayXd ret = f1*input.x;
      x->set(points[1]);
      ret -= f1*input.x;
      x->set(points[2]);
      ret -= f2*input.x;
      x->set(points[3]);
      ret += f2*input.x;
      x->set(x0);
       
      rets[0].x.block(i, 0, 1, size) = ret;
       std::cout <<"Jac at step " << i << " " << rets[0].mat << std::endl;
    }
       std::cout << "Final jac " << rets[0].x << std::endl;
}

void Jacobian::calcTypes(Atypes args, Rtypes rets){
    auto tmp = args[0];
    tmp.size();
    std::cout << "m_pars.size() * args[0].size() = " << m_pars.size() << " * " << args[0].size() << std::endl;
    rets[0] = DataType().points().shape(m_pars.size()*args[0].size());
}

template <typename T>
inline Eigen::ArrayXd Jacobian::computeDerivative(const T& input, Parameter<double>* x) {
  auto x0 = x->value();

  double f1 = 4.0/(3.0*m_reldelta);
  double f2 = 1.0/(6.0*m_reldelta);

  std::array<double, 4> points;
  points[0] = x->relativeValue(+m_reldelta/2);
  points[1] = x->relativeValue(-m_reldelta/2);
  points[2] = x->relativeValue(+m_reldelta);
  points[3] = x->relativeValue(-m_reldelta);

  x->set(points[0]);

  Eigen::ArrayXd ret = f1*input.x;
  x->set(points[1]);
  ret -= f1*input.x;
  x->set(points[2]);
  ret -= f2*input.x;
  x->set(points[3]);
  ret += f2*input.x;
  x->set(x0);

  return ret;
}



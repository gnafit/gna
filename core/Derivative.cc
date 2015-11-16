#include "Derivative.hh"

void Derivative::calcDerivative(Args args, Rets rets) {
  auto x0 = m_x->value();

  double f1 = 4.0/(3.0*m_reldelta);
  double f2 = 1.0/(6.0*m_reldelta);

  std::array<double, 4> points;
  points[0] = m_x->relativeValue(+m_reldelta/2);
  points[1] = m_x->relativeValue(-m_reldelta/2);
  points[2] = m_x->relativeValue(+m_reldelta);
  points[3] = m_x->relativeValue(-m_reldelta);

  m_x->set(points[0]);
  rets[0].x = f1*args[0].x;
  m_x->set(points[1]);
  rets[0].x -= f1*args[0].x;

  m_x->set(points[2]);
  rets[0].x -= f2*args[0].x;
  m_x->set(points[3]);
  rets[0].x += f2*args[0].x;

  m_x->set(x0);
}

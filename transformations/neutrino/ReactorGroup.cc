#include <array>

#include <boost/format.hpp>

#include "ReactorGroup.hh"

ReactorGroup::ReactorGroup(size_t count)
  : m_Ls(count), m_Ps(count)
{
  std::vector<changeable> deps;
  deps.reserve(2*count);
  for (size_t i = 0; i < count; ++i) {
    variable_(&m_Ps[i], (boost::format("P_%1%")%i).str());
    deps.push_back(m_Ps[i]);
    variable_(&m_Ls[i], (boost::format("L_%1%")%i).str());
    deps.push_back(m_Ls[i]);
  }
  m_Lavg = evaluable_<double>("Lavg", [this]() {
      double s1 = 0;
      double s2 = 0;
      for (size_t i = 0; i < m_Ls.size(); ++i) {
        double x = m_Ps[i]/std::pow(m_Ls[i], 2);
        s1 += x*m_Ls[i];
        s2 += x;
      }
      return s1/s2;
    }, deps);
  m_Pavg = evaluable_<double>("Pavg", [this]() {
      double s1 = 0;
      double s2 = 0;
      for (size_t i = 0; i < m_Ls.size(); ++i) {
        double x = m_Ps[i]/std::pow(m_Ls[i], 2);
        s1 += x*m_Ls[i];
        s2 += x;
      }
      return s1*s1/s2;
    }, deps);
  deps.push_back(m_Lavg);
  evaluable_<std::array<double, 3>>("weights", [this]() {
      std::array<double, 3> ret{{0.0, 0.0, 0.0}};
      double sum = 0.0;
      for (size_t i = 0; i < m_Ls.size(); ++i) {
        double lambda = m_Ls[i]/m_Lavg - 1;
        double lambda2 = pow(lambda, 2);
        double x = m_Ps[i]/std::pow(m_Ls[i], 2);
        sum += x;
        ret[0] += x*lambda2;
        ret[1] += x*lambda2*lambda;
        ret[2] += x*lambda2*lambda2;
      }
      for (double &r: ret) {
        r /= sum;
      }
      return ret;
    }, deps);
}

#include <boost/format.hpp>
#include <boost/math/constants/constants.hpp>

using boost::format;

#include <stdexcept>
#include <cstring>
#include <cstdlib>

#include "UncertainParameter.hh"

static double castfloat(const char *v) {
  const char *vend = v + strlen(v);
  char *endptr;
  double fv = strtod(v, &endptr);
  if (endptr != vend) {
    throw std::runtime_error(
      (format("invalid floating point value: `%1%'") % v).str()
      );
  }
  return fv;
}

double Parameter::cast(const std::string &v) const {
  return castfloat(v.c_str());
}

void UniformAngleParameter::set(double value) {
  const double pi = boost::math::constants::pi<double>();

  double v = value - static_cast<int>(value/pi)*2*pi;
  if (v > pi) {
    v -= 2*pi;
  }
  m_var = v;
}

double UniformAngleParameter::cast(const std::string &v) const {
  const double pi = boost::math::constants::pi<double>();

  const char *vend = v.c_str() + v.length();
  char *endptr;
  double fv = std::strtod(v.c_str(), &endptr);
  if (endptr == vend) {
    return fv;
  }
  const char *pipos = std::strstr(v.c_str(), "pi");
  if (endptr == pipos) {
    double d = strtod(pipos+2, &endptr);
    if (endptr == vend) {
      return fv / d * pi;
    }
  }
  throw std::runtime_error("invalid uniform angle value");
}

#include <stdexcept>

#include <boost/format.hpp>
using boost::format;

#include "Transformation.hh"

using std::string;

using Input = InputDescriptor;
using Output = OutputDescriptor;

Input InputDescriptor::invalid(const std::string name) {
  throw std::runtime_error(
    (format("Input: invalid entry, name == `%1%'") % name).str());
}

Output OutputDescriptor::invalid(const std::string name) {
  throw std::runtime_error(
    (format("Output: invalid entry, name == `%1%'") % name).str());
}

#include <utility>
#include <vector>

#include <boost/format.hpp>
using boost::format;

#include <cstdio>

#include "ParametersGroup.hh"

ParametersGroup::ParametersGroup(GNAObject *parent, Fields fields)
  : m_parent(parent), m_fields(std::move(fields))
{
}

void ParametersGroup::initFields(const std::vector<std::string> &params) {
  for (auto pname: params) {
    variable_(pname);
  }
}

void ParametersGroup::checkField(const std::string &name) {
  if (m_fields.count(name) == 0) {
    throw std::runtime_error(
      (format("Parametrized::Group: unknown parameter `%1%'") % name).str()
      );
  }
}

const std::string &ParametersGroup::fieldName(Field field) const {
  for (const auto &pair: m_fields) {
    if (std::get<0>(pair.second) == field) {
      return pair.first;
    }
  }
  throw std::runtime_error(
    (format("Parametrized::Group: unknown field `%1%'") % field).str()
    );
}

void ParametersGroup::dump() {
  for (const auto &pair: m_fields) {
    Field field = std::get<0>(pair.second);
    fprintf(stderr, "variable %s[%p]: %p\n",
            pair.first.c_str(), (void*)field, field->rawdata());
  }
}

ExpressionsProvider::ExpressionsProvider(ParametersGroup *pgroup)
  : m_pgroup(pgroup)
{
  pgroup->setExpressions(*this);
}

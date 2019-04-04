#include <utility>
#include <vector>
#include "fmt/format.h"
using fmt::format;

#include <cstdio>

#include "ParametersGroup.hh"

ParametersGroup::ParametersGroup(GNAObject *parent, Fields fields)
  : m_parent(dynamic_cast<ParametrizedTypes::ParametrizedBase*>(parent)), m_fields(std::move(fields))
{
}

void ParametersGroup::initFields(const std::vector<std::string> &params) {
  for (const auto& pname: params) {
    variable_(pname);
  }
}

void ParametersGroup::checkField(const std::string &name) {
  if (m_fields.count(name) == 0) {
    throw std::runtime_error(
      (fmt::format("Parametrized::Group: unknown parameter `{0}'", name))
      );
  }
}

const std::string &ParametersGroup::fieldName(Field field) const {
  for (const auto &pair: m_fields.expose()) {
    if (std::get<0>(pair.second) == field) {
      return pair.first;
    }
  }
  throw std::runtime_error(
    (fmt::format("Parametrized::Group: unknown field `{0}'", field->name() ))
    );
}

void ParametersGroup::dump() {
  for (const auto &pair: m_fields.expose()) {
    Field field = std::get<0>(pair.second);
    fmt::print(stderr, "variable {0}[{1}]: {2}\n", pair.first.c_str(), static_cast<void*>(field), field->rawdata());
    /* fprintf(stderr, "variable %s[%p]: %p\n",
     *         pair.first.c_str(), static_cast<void*>(field), field->rawdata()); */
  }
}

ExpressionsProvider::ExpressionsProvider(ParametersGroup *pgroup)
  : m_pgroup(pgroup)
{
  pgroup->setExpressions(*this);
}

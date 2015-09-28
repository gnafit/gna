#include <vector>

#include <boost/format.hpp>
using boost::format;

#include <cstdio>

#include "ParametersGroup.hh"

ParametersGroup::ParametersGroup(GNAObject *parent, const Fields &fields,
                                 const ExpressionsList &exprs)
  : m_parent(parent), m_fields(fields), m_exprs(exprs)
{
}

void ParametersGroup::initFields(const List &params) {
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
    if (pair.second == field) {
      return pair.first;
    }
  }
  throw std::runtime_error(
    (format("Parametrized::Group: unknown field `%1%'") % field).str()
    );
}

ParametersGroup::Handle ParametersGroup::variable_(const std::string &name) {
  checkField(name);
  if (m_fields[name]->isnull()) {
    return m_parent->variable_(m_fields[name], name);
  } else {
    return m_parent->getByField(m_fields[name]);
  }
}

void ParametersGroup::dump() {
  for (const auto &pair: m_fields) {
    fprintf(stderr, "variable %s[%p]: %p\n",
            pair.first.c_str(), (void*)pair.second, pair.second->rawdata());
  }
}

ExpressionsProvider::ExpressionsProvider(ParametersGroup *pgroup)
  : m_pgroup(pgroup)
{
  auto &exprs = pgroup->m_exprs;
  for (size_t i = 0; i < exprs.size(); ++i) {
    const ParametersGroup::Expression &expr = exprs[i];
    std::string name = pgroup->fieldName(expr.dest);
    std::vector<changeable> deps;
    for (ParametersGroup::Field f: expr.sources) {
      pgroup->variable_(pgroup->fieldName(f));
      deps.push_back(*f);
    }
    evaluable_(pgroup->fieldName(expr.dest), expr.func, deps);
  }
}

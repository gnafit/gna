#include "ParametrizedEntry.hh"

#include <stdexcept>

#include "fmt/format.h"
#include "ParametrizedBase.hh"
#include "VariableHandle.hh"
#include "EvaluableHandle.hh"
#include "EvaluableEntry.hh"

using ParametrizedTypes::ParametrizedEntry;
using ParametrizedTypes::VariableHandle;
using ParametrizedTypes::EvaluableEntry;
using ParametrizedTypes::EvaluableHandle;
using ParametrizedTypes::ParametrizedBase;

ParametrizedEntry::ParametrizedEntry(parameter<void> par, std::string name, const ParametrizedBase *parent)
  : name(std::move(name)), required(true),
    par(par), var(par), state(Free), claimer(nullptr),
    parent(parent)
{
}

ParametrizedEntry::ParametrizedEntry(const ParametrizedEntry &other, const ParametrizedBase *parent)
  : ParametrizedEntry(other.par, other.name, parent)
{
}

void ParametrizedEntry::bind(variable<void> newvar) {
  if (!var.is(par)) {
    throw std::runtime_error(
      fmt::format("can not rebind parameter `{0}'", name)
      );
  }
  var.replace(newvar);
  par.replace(newvar);
  if (field != &var) {
    field->assign(newvar);
  }
  state = ParametrizedEntry::State::Bound;
}

parameter<void> ParametrizedEntry::claim(ParametrizedBase *other) {
  if (state != ParametrizedEntry::State::Free) {
    throw std::runtime_error(
      fmt::format("claiming non-free parameter `{0}'", name)
      );
  }
  state = ParametrizedEntry::State::Claimed;
  claimer = other;
  return par;
}



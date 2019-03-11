#include "EvaluableEntry.hh"

using ParametrizedTypes::EvaluableEntry;

EvaluableEntry::EvaluableEntry(std::string name,
                               const SourcesContainer &sources,
                               dependant<void> dependant,
                               const ParametrizedBase *parent)
  : name(std::move(name)), sources(sources), dep(dependant), parent(parent)
{
}

EvaluableEntry::EvaluableEntry(const EvaluableEntry &other,
                               const ParametrizedBase *parent)
  : EvaluableEntry(other.name, other.sources, other.dep, parent)
{
}

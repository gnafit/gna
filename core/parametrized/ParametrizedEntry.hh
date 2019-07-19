#pragma once

#include <string>
#include <boost/noncopyable.hpp>

#include "variable.hh"
#include "parameter.hh"

namespace GNA{
  namespace GNAObjectTemplates{
    template<typename FloatType>
    class ParametersGroupT;
  }
}

template <typename SourceFloatType,typename SinkFloatType>
class GNAObjectT;

namespace ParametrizedTypes {
  class ParametrizedBase;
  class ParametrizedEntry: public boost::noncopyable {
    friend class ParametrizedBase;
  public:
    enum State {
      Free = 0,   ///< Default state
      Claimed,    ///< The parameter access is claimed
      Bound,      ///< The Entry is a view to another Entry
    };

    ParametrizedEntry(parameter<void> par, std::string name, const ParametrizedBase *parent);
    ParametrizedEntry(const ParametrizedEntry &other, const ParametrizedBase *parent);

    void bind(variable<void> var);
    parameter<void> claim(ParametrizedBase *other);

    std::string name;
    bool required;
    parameter<void> par;               ///< Parameter view (able to set the value)
    variable<void> var;                ///< Varaible view (unable to set the value)
    State state;
    void *claimer;                     ///< An object (ParametrizedBase), that claimed the parameter. FIXME: not used
    variable<void> *field;             ///< Reference to the transformation member variable (GNAObject)
    const ParametrizedBase *parent;
  };
}

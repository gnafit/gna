#pragma once

#include <string>
#include <boost/noncopyable.hpp>

#include "variable.hh"
#include "parameter.hh"

class ParametersGroup;
template <typename SourceFloatType,typename SinkFloatType>
class GNAObjectT;

namespace ParametrizedTypes {
  class ParametrizedBase;
  class ParametrizedEntry: public boost::noncopyable {
    friend class ParametrizedBase;
  public:
    enum State {
      Free = 0, Claimed, Bound,
    };

    ParametrizedEntry(parameter<void> par, std::string name, const ParametrizedBase *parent);
    ParametrizedEntry(const ParametrizedEntry &other, const ParametrizedBase *parent);

    void bind(variable<void> var);
    parameter<void> claim(ParametrizedBase *other);

    std::string name;
    bool required;
    parameter<void> par;
    variable<void> var;
    State state;
    void *claimer;
    variable<void> *field;
    const ParametrizedBase *parent;
  };
}

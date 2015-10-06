#ifndef PARAMETERSGROUP_H
#define PARAMETERSGROUP_H

#include <functional>
#include <vector>
#include <map>

#include "GNAObject.hh"

class ExpressionsProvider;
class ParametersGroup {
  friend class ExpressionsProvider;
  friend class Reparametrizer;
protected:
  typedef ParametrizedTypes::VariableHandle<void> Handle;
  typedef variable<void> *Field;
  typedef Handle(ParametersGroup::*Factory)(Field, const std::string&);
  typedef std::vector<Field> FieldsVector;
  friend class Fields;
  class Fields: public std::map<std::string, std::tuple<Field, Factory>> {
  public:
    template <typename T>
    Fields &add(variable<T> *field, const std::string &name) {
      (*this)[name] = std::make_tuple(field, &ParametersGroup::factory<T>);
      return *this;
    }
  };
public:
  ParametersGroup(GNAObject *parent, const Fields &fields);
  virtual ~ParametersGroup() { }

  void dump();

  template <typename T>
  Handle factory(Field field, const std::string &name) {
    return m_parent->variable_(static_cast<variable<T>*>(field), name);
  }

  template <typename T=void>
  ParametrizedTypes::VariableHandle<T> variable_(const std::string &name) {
    using namespace ParametrizedTypes;
    checkField(name);
    Field field;
    Factory factory;
    std::tie(field, factory) = m_fields[name];
    if (field->isnull()) {
      return VariableHandle<T>((this->*factory)(field, name));
    } else {
      return VariableHandle<T>(m_parent->getByField(field));
    }
  }
  template <typename T>
  ParametrizedTypes::VariableHandle<T> variable_(variable<T> *field) {
    return variable_<T>(fieldName(field));
  }
protected:
  void initFields(const std::vector<std::string> &params);
  void checkField(const std::string &name);
  const std::string &fieldName(Field field) const;
  virtual void setExpressions(ExpressionsProvider &/*provider*/) { }

  GNAObject *m_parent;
  Fields m_fields;
};

class ExpressionsProvider: public GNAObject {
public:
  ~ExpressionsProvider() { delete m_pgroup; }

  template <typename T, typename FuncType>
  ExpressionsProvider &add(variable<T> *field,
                           const ParametersGroup::FieldsVector &sources,
                           FuncType func) {
    std::string name = m_pgroup->fieldName(field);
    std::vector<changeable> deps;
    for (ParametersGroup::Field f: sources) {
      m_pgroup->variable_(m_pgroup->fieldName(f)).required(false);
      deps.push_back(*f);
    }
    evaluable_(name, std::function<T()>(func), deps);
    return *this;
  }
protected:
  ExpressionsProvider(ParametersGroup *pgroup);

  ParametersGroup *m_pgroup;

  ClassDef(ExpressionsProvider, 0);
};

#endif // PARAMETERSGROUP_H

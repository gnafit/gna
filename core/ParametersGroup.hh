#ifndef PARAMETERSGROUP_H
#define PARAMETERSGROUP_H

#include <functional>
#include <vector>
#include <map>

#include "GNAObject.hh"

class ParametersGroup {
  friend class ExpressionsProvider;
  friend class Reparametrizer;
protected:
  typedef variable<double> *Field;
  typedef std::map<std::string, Field> Fields;
  typedef std::vector<std::string> List;
  typedef std::vector<Field> FieldsList;
  typedef ParametrizedTypes::VariableHandle Handle;
  struct Expression {
    Field dest;
    FieldsList sources;
    std::function<double()> func;
  };
  typedef std::vector<Expression> ExpressionsList;
public:
  ParametersGroup(GNAObject *parent, const Fields &fields,
                  const ExpressionsList &m_exprs);
  virtual ~ParametersGroup() { }

  void dump();

  Handle variable_(const std::string &name);
protected:
  void initFields(const List &params);
  void checkField(const std::string &name);
  const std::string &fieldName(Field field) const;

  GNAObject *m_parent;
  Fields m_fields;
  ExpressionsList m_exprs;
};

class ExpressionsProvider: public GNAObject {
public:
  ~ExpressionsProvider() { delete m_pgroup; }
protected:
  ExpressionsProvider(ParametersGroup *pgroup);

  ParametersGroup *m_pgroup;

  ClassDef(ExpressionsProvider, 0);
};

#endif // PARAMETERSGROUP_H

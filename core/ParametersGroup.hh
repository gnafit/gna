#pragma once

#include <functional>
#include <vector>
#include <map>

#include "GNAObject.hh"

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    class ExpressionsProviderT;

    template<typename FloatType>
    class ParametersGroupT {
      template<typename FloatTypeB>
      friend class GNA::GNAObjectTemplates::ExpressionsProviderT;

    protected:
      using ParametersGroup = ParametersGroupT<FloatType>;
      using ExpressionsProvider = ExpressionsProviderT<FloatType>;
      using Handle = ParametrizedTypes::VariableHandle<void>;
      using Field = variable<void>*;
      using Factory = Handle (ParametersGroup::*)(Field, const std::string &);
      using FieldsVector =  std::vector<Field>;
      friend class Fields;

      class Fields  {
      public:
        template <typename T>
        Fields &add(variable<T> *field, const std::string &name) {
          m_map[name] = std::make_tuple(field, &ParametersGroup::factory<T>);
          return *this;
        }

      using FieldsStorage = std::map<std::string, std::tuple<Field, Factory>>;
      const FieldsStorage& expose() const noexcept {return m_map;};
      size_t count(std::string entry) const noexcept {return m_map.count(entry);};
      typename FieldsStorage::value_type::second_type& operator[](std::string mem) noexcept {return m_map[mem];};
      private:
        FieldsStorage m_map;
      };

    public:
      ParametersGroupT(GNAObjectT<FloatType,FloatType> *parent, Fields fields);
      virtual ~ParametersGroupT() = default;

      void dump();

      template <typename T>
      Handle factory(Field field, const std::string &name) {
        return m_parent->variable_(static_cast<variable<T>*>(field), name, 0u);
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
      const std::string &fieldName(Field field) const {
        for (const auto &pair: m_fields.expose()) {
          if (std::get<0>(pair.second) == field) {
            return pair.first;
          }
        }
        throw std::runtime_error(
          (fmt::format("Parametrized::Group: unknown field `{0}'", field->name() ))
          );
      }
      virtual void setExpressions(ExpressionsProvider &/*provider*/) { }

      GNAObjectT<FloatType,FloatType>* m_parent;
      Fields m_fields;
    };

    template<typename FloatType>
    class ExpressionsProviderT: public GNAObjectT<FloatType,FloatType> {
      template<typename FloatTypeB>
      friend class GNA::GNAObjectTemplates::ParametersGroupT;
    public:
      using ExpressionsProvider = ExpressionsProviderT<FloatType>;
      using ParametersGroup = ParametersGroupT<FloatType>;
      using GNAObject = GNAObjectT<FloatType,FloatType>;
      using GNAObject::ParametrizedBase::evaluable_;

      ~ExpressionsProviderT() { delete m_pgroup; }

      template <typename T, typename FuncType>
      ExpressionsProvider &add(variable<T> *field,
                               const typename ParametersGroup::FieldsVector &sources,
                               FuncType func) {
        std::string name = m_pgroup->fieldName(field);
        std::vector<changeable> deps;
        for (typename ParametersGroup::Field f: sources) {
          m_pgroup->variable_(m_pgroup->fieldName(f)).required(false);
          deps.push_back(*f);
        }
        this->evaluable_(name, std::function<T()>(func), deps);
        return *this;
      }

      template <typename T, typename FuncType>
      ExpressionsProvider &add(variable<T> *field,
                               const typename ParametersGroup::FieldsVector &sources,
                               FuncType func, size_t size) {
        std::string name = m_pgroup->fieldName(field);
        std::vector<changeable> deps;
        for (typename ParametersGroup::Field f: sources) {
          m_pgroup->variable_(m_pgroup->fieldName(f)).required(false);
          deps.push_back(*f);
        }
        this->evaluable_(name, size, std::function<void(arrayview<T>&)>(func), deps);
        return *this;
      }
    protected:
      ExpressionsProviderT(ParametersGroup *pgroup);

      ParametersGroup *m_pgroup;
    };
  }
}


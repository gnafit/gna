#include <utility>
#include <vector>
#include "fmt/format.h"
using fmt::format;

#include <cstdio>

#include "ParametersGroup.hh"

namespace GNA {
  namespace GNAObjectTemplates {
    template<typename FloatType>
    ParametersGroupT<FloatType>::ParametersGroupT(GNAObjectT<FloatType,FloatType> *parent, Fields fields)
      : m_parent(parent), m_fields(std::move(fields))
    {
    }

    template<typename FloatType>
    void ParametersGroupT<FloatType>::initFields(const std::vector<std::string> &params) {
      for (const auto& pname: params) {
        variable_(pname);
      }
    }

    template<typename FloatType>
    void ParametersGroupT<FloatType>::checkField(const std::string &name) {
      if (m_fields.count(name) == 0) {
        throw std::runtime_error(
          (fmt::format("Parametrized::Group: unknown parameter `{0}'", name))
          );
      }
    }

    //template<typename FloatType>
    //const std::string &ParametersGroupT<FloatType>::fieldName(ParametersGroupT<FloatType>::Field field) const {
      //for (const auto &pair: m_fields.expose()) {
        //if (std::get<0>(pair.second) == field) {
          //return pair.first;
        //}
      //}
      //throw std::runtime_error(
        //(fmt::format("Parametrized::Group: unknown field `{0}'", field->name() ))
        //);
    //}

    template<typename FloatType>
    void ParametersGroupT<FloatType>::dump() {
      for (const auto &pair: m_fields.expose()) {
        Field field = std::get<0>(pair.second);
        fmt::print(stderr, "variable {0}[{1}]: {2}\n", pair.first.c_str(), static_cast<void*>(field), field->rawdata());
        /* fprintf(stderr, "variable %s[%p]: %p\n",
         *         pair.first.c_str(), static_cast<void*>(field), field->rawdata()); */
      }
    }

    template<typename FloatType>
    ExpressionsProviderT<FloatType>::ExpressionsProviderT(ParametersGroupT<FloatType> *pgroup)
      : m_pgroup(pgroup)
    {
      pgroup->setExpressions(*this);
    }
  }
}

template class GNA::GNAObjectTemplates::ExpressionsProviderT<double>;
template class GNA::GNAObjectTemplates::ParametersGroupT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::ExpressionsProviderT<float>;
  template class GNA::GNAObjectTemplates::ParametersGroupT<float>;
#endif

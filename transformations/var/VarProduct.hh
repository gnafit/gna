#pragma once

#include "GNAObject.hh"

namespace GNA{
    namespace GNAObjectTemplates{
       /**
        * @brief Transformation implementing the evaluable for variable difference 'a*b*c*...'.
        *
        * For at least two input variables computes the product 'a*b*c*...'.
        *
        * @author Maxim Gonchar
        * @date 02.2018
        */
        template<typename FloatType>
        class VarProductT: public GNAObjectT<FloatType,FloatType> {
        public:
            VarProductT(const std::vector<std::string>& varnames, const std::string& productname); ///< Constructor.

        protected:
            std::vector<variable<FloatType>> m_vars; ///< List of variables.
            dependant<FloatType> m_product;          ///< The product result.
        };
  }
}

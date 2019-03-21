#pragma once

#include "GNAObject.hh"

namespace GNA{
    namespace GNAObjectTemplates{
        /**
         * @brief Transformation implementing the evaluable for variable sum 'a+b+c+...'.
         *
         * For at least two input variables computes the sum 'a+b+c+...'.
         *
         * @author Maxim Gonchar
         * @date 02.2018
         */
        template<typename FloatType>
        class VarSumT: public GNAObjectT<FloatType,FloatType> {
        public:
            VarSumT(const std::vector<std::string>& varnames, const std::string& sumname); ///< Constructor

        protected:
            std::vector<variable<FloatType>> m_vars; ///< List of variables to sum.
            dependant<FloatType> m_sum;              ///< The sum result.
        };
    }
}

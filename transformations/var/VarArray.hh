#pragma once

#include "GNAObject.hh"

namespace GNA{
    namespace GNAObjectTemplates{
        template<typename FloatType>
        class VarArrayT: public GNASingleObjectT<FloatType,FloatType>,
                         public TransformationBind<VarArrayT<FloatType>,FloatType,FloatType> {
        private:
            using BaseClass = GNASingleObjectT<FloatType,FloatType>;
        public:
            using typename BaseClass::FunctionArgs;
            using typename BaseClass::TypesFunctionArgs;
            using VarArrayType = VarArrayT<FloatType>;

        public:
            VarArrayT(const std::vector<std::string>& varnames); ///< Constructor.

        protected:
            void typesFunction(TypesFunctionArgs& fargs);
            void function(FunctionArgs& fargs);

            std::vector<variable<FloatType>> m_vars;            ///< List of variables.
        };
    }
}

#pragma once

#include "GNAObject.hh"

template<typename FloatType>
class arrayviewAllocator;

namespace GNA{
    namespace GNAObjectTemplates{
        template<typename FloatType>
        class VarArrayPreallocatedT: public GNASingleObjectT<FloatType,FloatType>,
                                     public TransformationBind<VarArrayPreallocatedT<FloatType>,FloatType,FloatType> {
        private:
            using BaseClass = ::GNASingleObjectT<FloatType,FloatType>;
            using AllocatorType = ::arrayviewAllocator<FloatType>;
        public:
            using typename BaseClass::FunctionArgs;
            using typename BaseClass::TypesFunctionArgs;
            using VarArrayPreallocatedType = VarArrayPreallocatedT<FloatType>;

        public:
            //VarArrayPreallocatedT(const std::vector<std::string>& varnames);     ///< Constructor.
            VarArrayPreallocatedT(const std::vector<variable<FloatType>>& vars); ///< Constructor.

        protected:
            void initTransformation();

            void typesFunction(TypesFunctionArgs& fargs);
            void function(FunctionArgs& fargs);

            std::vector<variable<FloatType>> m_vars;        ///< List of variables.
            std::vector<variable<FloatType>> m_dependants;  ///< List of evaluables.

            AllocatorType* m_allocator=nullptr;
        };
    }
}

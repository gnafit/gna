#pragma once

#include <string>
#include <vector>

#include "GNAObject.hh"

namespace GNA{
    namespace GNAObjectTemplates {
        template<typename FloatType>
        class PolyRatioT: public GNASingleObjectT<FloatType,FloatType>,
                          public TransformationBind<PolyRatioT<FloatType>,FloatType,FloatType> {
        protected:
            using PolyRatioType = PolyRatioT<FloatType>;
            using GNAObjectType = GNAObjectT<FloatType,FloatType>;
        public:
            using typename GNAObjectType::FunctionArgs;

            PolyRatioT(const std::vector<std::string> &nominator, const std::vector<std::string> &denominator);

        protected:
            void polyratio(FunctionArgs& fargs);

            std::vector<variable<FloatType>> m_weights_nominator;
            std::vector<variable<FloatType>> m_weights_denominator;
        };
    }
}

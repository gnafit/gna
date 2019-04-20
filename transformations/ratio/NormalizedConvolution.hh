#pragma once

#include <string>
#include <vector>
#include <boost/optional.hpp>

#include "GNAObject.hh"

namespace GNA{
    namespace GNAObjectTemplates {
        template<typename FloatType>
        class NormalizedConvolutionT: public GNASingleObjectT<FloatType,FloatType>,
                                      public TransformationBind<NormalizedConvolutionT<FloatType>,FloatType,FloatType> {
        protected:
            using NormalizedConvolutionType = NormalizedConvolutionT<FloatType>;
            using GNAObjectType = GNAObjectT<FloatType,FloatType>;
        public:
            using typename GNAObjectType::FunctionArgs;

            NormalizedConvolutionT();
            NormalizedConvolutionT(const std::string& scale);

        protected:
            void convolute(FunctionArgs& fargs);

            boost::optional<variable<FloatType>> m_scale;
        };
    }
}

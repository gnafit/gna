#pragma once

#include <string>
#include <vector>
#include <boost/optional.hpp>

#include "GNAObject.hh"

namespace GNA{
    namespace GNAObjectTemplates {
        template<typename FloatType>
        class ConvolutionT: public GNASingleObjectT<FloatType,FloatType>,
                            public TransformationBind<ConvolutionT<FloatType>,FloatType,FloatType> {
        protected:
            using ConvolutionType = ConvolutionT<FloatType>;
            using GNAObjectType = GNAObjectT<FloatType,FloatType>;
        public:
            using typename GNAObjectType::FunctionArgs;

            ConvolutionT();
            ConvolutionT(const std::string& scale);

        protected:
            void convolute(FunctionArgs& fargs);

            boost::optional<variable<FloatType>> m_scale;
        };
    }
}

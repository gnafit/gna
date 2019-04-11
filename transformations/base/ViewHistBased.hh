#pragma once

#include <vector>
#include "GNAObject.hh"
#include "GNAObjectBind1N.hh"
#include <boost/optional.hpp>

namespace GNA{
    namespace GNAObjectTemplates{
        template<typename FloatType>
        class ViewHistBasedT: public GNAObjectBind1N<FloatType>,
                              public TransformationBind<ViewHistBasedT<FloatType>,FloatType,FloatType> {
        private:
            using BaseClass = GNAObjectT<FloatType,FloatType>;
            using typename BaseClass::TypesFunctionArgs;
            using typename BaseClass::FunctionArgs;
        public:
            using ViewHistBasedType = ViewHistBasedT<FloatType>;
            using typename BaseClass::SingleOutput;
            using TransformationDescriptor = typename BaseClass::TransformationDescriptorType;
            using OutputDescriptor = typename BaseClass::OutputDescriptor;

            ViewHistBasedT(FloatType threshold, FloatType ceiling);
            ViewHistBasedT(SingleOutput& output, FloatType threshold, FloatType ceiling);

            void set(SingleOutput& hist){
                hist.single() >> this->transformations.front().inputs.front();
            }

            TransformationDescriptor add_transformation(const std::string& name="");

        protected:
            void histTypes(TypesFunctionArgs& fargs);
            void types(TypesFunctionArgs& fargs);
            void init();

            boost::optional<FloatType> m_threshold;
            boost::optional<FloatType> m_ceiling;

            boost::optional<size_t> m_start;
            boost::optional<size_t> m_len;
            size_t m_full_length;
        };
    }
}


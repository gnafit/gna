#pragma once

#include <vector>
#include "GNAObject.hh"
#include <boost/optional.hpp>

namespace GNA{
    namespace GNAObjectTemplates{
        template<typename FloatType>
        class ViewRearT: public GNAObjectT<FloatType,FloatType>,
                         public TransformationBind<ViewRearT<FloatType>,FloatType,FloatType> {
        private:
            using BaseClass = GNAObjectT<FloatType,FloatType>;
            using typename BaseClass::TypesFunctionArgs;
            using typename BaseClass::FunctionArgs;
        public:
            using ViewRearType = ViewRearT<FloatType>;
            using typename BaseClass::SingleOutput;
            using DataPtr = std::unique_ptr<Data<FloatType>>;

            ViewRearT(size_t start, size_t len);
            ViewRearT(SingleOutput* output, size_t start, size_t len);
            ViewRearT(size_t start, size_t len, FloatType fill_value);
            ViewRearT(SingleOutput* output, size_t start, size_t len, FloatType fill_value);

            void allowThreshold() { m_threshold_forbidden=false; }
        protected:
            void types(TypesFunctionArgs& fargs);
            void init();
            bool dtypeInconsistent(const DataType& input, const DataType& required);

            size_t m_threshold_forbidden=true;
            size_t m_start=0lu;
            size_t m_len;

            boost::optional<FloatType> m_fill_value;

            DataPtr m_data;
        };
    }
}


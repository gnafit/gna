#pragma once

#include <vector>
#include "GNAObject.hh"

namespace GNA{
    namespace GNAObjectTemplates{
        /**
         * @brief Transformation object holding a static 1- or 2-dimensional array.
         *
         * Outputs:
         *   - `points.points` - 1- or 2- dimensional array with fixed data.
         *
         * @author Dmitry Taychenachev
         * @date 2015
         */
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

            const DataType& getDataType() { return m_datatype_sub; }
        protected:
            void types(TypesFunctionArgs& fargs);
            void init();

            size_t m_start;
            size_t m_len;

            DataType m_datatype_sub;
            DataPtr m_data;
        };
    }
}


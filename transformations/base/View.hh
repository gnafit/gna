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
        class ViewT: public GNAObjectT<FloatType,FloatType>,
                     public TransformationBind<ViewT<FloatType>,FloatType,FloatType> {
        private:
        public:
            using BaseClass = GNAObjectT<FloatType,FloatType>;
            using typename BaseClass::TypesFunctionArgs;
            using typename BaseClass::FunctionArgs;

            using ViewType = ViewT<FloatType>;
            using typename BaseClass::SingleOutput;

            ViewT(size_t start, size_t len);
            ViewT(SingleOutput* output, size_t start, size_t len);

        protected:
            void types(TypesFunctionArgs& fargs);
            void init();

            size_t m_start;
            size_t m_len;
        };
    }
}


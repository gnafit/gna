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

            ViewRearT()                                                                     : ViewRearT(true)                                                { }
            ViewRearT(size_t start)                                                         : ViewRearT(true, nullptr, start)                                { }
            ViewRearT(size_t start, size_t len)                                             : ViewRearT(true, nullptr, start, len)                           { }
            ViewRearT(FloatType fill_value)                                                 : ViewRearT(true, nullptr, boost::none, boost::none, fill_value) { }
            ViewRearT(size_t start, FloatType fill_value)                                   : ViewRearT(true, nullptr, start, boost::none, fill_value)       { }
            ViewRearT(size_t start, size_t len, FloatType fill_value)                       : ViewRearT(true, nullptr, start, len, fill_value)               { }
            ViewRearT(SingleOutput& output)                                                 : ViewRearT(true, &output)                                       { }
            ViewRearT(SingleOutput& output, size_t start)                                   : ViewRearT(true, &output, start)                                { }
            ViewRearT(SingleOutput& output, size_t start, size_t len)                       : ViewRearT(true, &output, start, len)                           { }
            ViewRearT(SingleOutput& output, FloatType fill_value)                           : ViewRearT(true, &output, boost::none, boost::none, fill_value) { }
            ViewRearT(SingleOutput& output, size_t start, FloatType fill_value)             : ViewRearT(true, &output, start, boost::none, fill_value)       { }
            ViewRearT(SingleOutput& output, size_t start, size_t len, FloatType fill_value) : ViewRearT(true, &output, start, len, fill_value)               { }

            void determineOffset(SingleOutput& main, SingleOutput& sub, bool points_for_edges=false);
            void set(SingleOutput& output) { output.single() >> this->transformations.front().inputs.front(); }

            void allowThreshold() { m_threshold_forbidden=false; }

            size_t getStart() { return m_start ? m_start.value() : -1lu; }
        protected:
            ViewRearT(bool, // used exclusively for overloading specification
                      SingleOutput* output=nullptr,
                      boost::optional<size_t> start=boost::none,
                      boost::optional<size_t> len=boost::none,
                      boost::optional<FloatType> fill_value=boost::none
                      ) : m_start(start), m_len(len), m_fill_value(fill_value)
                {
                    init();
                    if(output){
                    set(*output);
                }
            }

            void offsetTypes(TypesFunctionArgs& fargs);
            void types(TypesFunctionArgs& fargs);
            void init();
            bool dtypeInconsistent(const DataType& input, const DataType& required);
            void determineStartLen(const DataType& main, const DataType& sub, TypesFunctionArgs& fargs);

            size_t m_threshold_forbidden=true;
            boost::optional<size_t> m_start;
            boost::optional<size_t> m_len;

            boost::optional<FloatType> m_fill_value;

            DataPtr m_data;

            bool m_auto_offset=false;
            bool m_points_for_edges=false;
        };
    }
}


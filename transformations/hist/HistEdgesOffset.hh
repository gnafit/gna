#pragma once

#include <vector>
#include <boost/optional.hpp>
#include "GNAObject.hh"

class HistEdgesOffset: public GNASingleObject,
                       public TransformationBind<HistEdgesOffset> {
public:
    HistEdgesOffset(size_t offset) : m_offset(offset) { init(); }
    HistEdgesOffset(size_t offset, double threshold) : m_offset(offset), m_threshold(threshold) { init(); }
    HistEdgesOffset(double threshold) : m_threshold(threshold) { init(); }

    HistEdgesOffset(SingleOutput& out, size_t offset) : HistEdgesOffset(offset) { add_input(out); }
    HistEdgesOffset(SingleOutput& out, size_t offset, double threshold) : HistEdgesOffset(offset, threshold) { add_input(out); }
    HistEdgesOffset(SingleOutput& out, double threshold) : HistEdgesOffset(threshold) { add_input(out); }
    void add_input(SingleOutput& out){ out.single() >> transformations.front().inputs.front(); }

    bool floatThreshold() { return bool(m_threshold); }
private:
    void init();
    void types(TypesFunctionArgs& fargs);
    void func(FunctionArgs& fargs);

    boost::optional<size_t> m_offset;
    boost::optional<double> m_threshold;
};

#pragma once

#include <vector>
#include <boost/optional.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include "GNAObject.hh"

class HistEdgesOffset: public GNAObject,
                       public TransformationBind<HistEdgesOffset> {
public:
    HistEdgesOffset(size_t offset) : m_offset(offset) { init(); }
    HistEdgesOffset(size_t offset, double threshold) : m_offset(offset), m_threshold(threshold) { init(); }
    HistEdgesOffset(double threshold) : m_threshold(threshold) { init(); }

    HistEdgesOffset(SingleOutput& out, size_t offset) : HistEdgesOffset(offset) { set(out); }
    HistEdgesOffset(SingleOutput& out, size_t offset, double threshold) : HistEdgesOffset(offset, threshold) { set(out); }
    HistEdgesOffset(SingleOutput& out, double threshold) : HistEdgesOffset(threshold) { set(out); }
    void set(SingleOutput& out){ out.single() >> transformations.front().inputs.front(); }

    //void add_transformation(double fill_value);
    //void add_input();

    bool floatThreshold() { return bool(m_threshold); }
    size_t getOffset() { return m_offset.value(); }
private:
    void init();
    void types(TypesFunctionArgs& fargs);
    void func(FunctionArgs& fargs);

    //void viewTypes(TypesFunctionArgs& fargs);
    //void viewFunc(FunctionArgs& fargs);

    boost::optional<size_t> m_offset;
    boost::optional<double> m_threshold;

    //boost::optional<double> m_fillvalue;

    //DataType m_dt_hist_input;
    //DataType m_dt_points;
    //DataType m_dt_points_truncated;
    //DataType m_dt_hist_truncated;
    //DataType m_dt_hist_threshold;
};

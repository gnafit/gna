#pragma once

#include "Statistic.hh"
#include "TransformationDescriptor.hh"

class StatisticOutput: public Statistic {
public:
    StatisticOutput(OutputDescriptor& output);
    virtual ~StatisticOutput() = default;

    double value() override { return m_output.data()[0]; }
private:
    void validate() const;

    OutputDescriptor m_output;
};


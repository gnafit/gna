#include "StatisticOutput.hh"

StatisticOutput::StatisticOutput(OutputDescriptor& output) : m_output(output) {
	validate();
}

void StatisticOutput::validate() const {
    auto& dt=m_output.datatype();
	if(dt.size()!=1u){
		throw std::runtime_error("Statistic is expected to have size=1");
	}
}

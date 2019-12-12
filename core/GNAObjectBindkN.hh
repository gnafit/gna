#pragma once

#include "GNAObject.hh"
#include <vector>

class GNAObjectBindkN: public GNAObjectT<double,double> {
public:
    GNAObjectBindkN(const std::string& transformation,
                    const std::vector<std::string>& inputs,
                    const std::string& output,
                    size_t transformation_offsset, size_t input_offset, size_t output_offset);

    virtual TransformationDescriptor add_transformation(const std::string& name="");

    void add_inputs(const std::vector<std::string>& inames={});
    void add_inputs(const SingleOutputsContainer& outputs);

    void add_inputs(SingleOutput* output)                                                { return add_inputs(SingleOutputsContainer({output})); }
    void add_inputs(SingleOutput* output1, SingleOutput* output2)                        { return add_inputs(SingleOutputsContainer({output1, output2})); }
    void add_inputs(SingleOutput* output1, SingleOutput* output2, SingleOutput* output3) { return add_inputs(SingleOutputsContainer({output1, output2, output3})); }

protected:
    void set_transformation_offset(size_t offset) { m_transformation_offset=offset; }
    void set_input_offset(size_t offset)          { m_input_offset=offset; }
    void set_output_offset(size_t offset)         { m_output_offset=offset; }

    std::string new_transformation_name(const std::string& name);

    void bind_tfirst_tlast(size_t noutput, size_t ninput);
    void set_open_input()   { m_open_inputs=m_input_names.size(); }
    void reset_open_input() { m_open_inputs=0u; }

private:
    std::string new_name(const std::string& base, size_t num, size_t offset, const std::string& altname="", size_t divisor=1);
    void add_inputs_only(const std::vector<std::string>& inames={});
    void add_inputs_only(const SingleOutputsContainer& outputs);
    InputDescriptor add_input(const std::string& iname="");
    void add_input(SingleOutput& output, const std::string& iname="");
    bool needs_output();
    OutputDescriptor add_output(const std::string& oname="");

    std::string m_transformation_name="";
    std::vector<std::string> m_input_names;
    std::string m_output_name="";

    size_t m_transformation_offset=0u;
    size_t m_input_offset=0u;
    size_t m_output_offset=0u;

    size_t m_open_inputs=0u;
};

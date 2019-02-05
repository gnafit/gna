#pragma once

#include "GNAObject.hh"

class GNAObjectBindMN: public GNAObject {
public:
    GNAObjectBindMN(const std::string& transformation, const std::string& input, const std::string& output);

    virtual TransformationDescriptor add_transformation(const std::string& name="");
    InputDescriptor  add_input(const std::string& iname="");
    OutputDescriptor add_input(SingleOutput& output, const std::string& iname="");
    OutputDescriptor add_output(const std::string& oname="");

    void add_inputs(const OutputDescriptor::OutputDescriptors& outputs);

protected:
    std::string new_transformation_name(const std::string& name);

    void set_open_input() { m_open_input=true; }

private:
    std::string new_name(const std::string& base, size_t num, const std::string& altname="");

    std::string m_transformation_name="";
    std::string m_input_name="";
    std::string m_output_name="";

    bool m_open_input=false;
};

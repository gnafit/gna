#pragma once

#include "GNAObject.hh"

template<typename FloatType>
class GNAObjectBindMN: public GNAObjectT<FloatType,FloatType> {
protected:
    using GNAObject = GNAObjectT<FloatType,FloatType>;
    using GNAObject::transformations;
public:
    using typename GNAObject::SingleOutput;
    using typename GNAObject::OutputDescriptor;
    using typename GNAObject::InputDescriptor;
    using TransformationDescriptor = typename GNAObject::TransformationDescriptorType;
    using OutputDescriptors = typename OutputDescriptor::OutputDescriptors;

    GNAObjectBindMN(const std::string& transformation, const std::string& input, const std::string& output);

    virtual TransformationDescriptor add_transformation(const std::string& name="");
    InputDescriptor  add_input(const std::string& iname="");
    OutputDescriptor add_input(SingleOutput& output, const std::string& iname="");
    OutputDescriptor add_output(const std::string& oname="");

    void add_inputs(const OutputDescriptors& outputs);

protected:
    std::string new_transformation_name(const std::string& name);

    void set_open_input() { m_open_input=true; }
    void reset_open_input() { m_open_input=false; }

private:
    std::string new_name(const std::string& base, size_t num, const std::string& altname="");

    std::string m_transformation_name="";
    std::string m_input_name="";
    std::string m_output_name="";

    bool m_open_input=false;
};

#pragma once

#include "GNAObject.hh"

/**
 * @brief Helper methods for GNAObject containing 1+N transformations
 *
 * Implements methods for transformations, contatining 1 common input and N other inputs.
 * For example matrix multiplication of form of:
 * R_i = M I_i
 *
 * where M is common matrix, passed as first input and applied to all the other inputs.
 *
 * @author Maxim Gonchar
 * @date 12.2018
 */
template<typename FloatType>
class GNAObjectBind1N: public GNAObjectT<FloatType,FloatType> {
protected:
    using GNAObject = GNAObjectT<FloatType,FloatType>;
    using GNAObject::transformations;
public:
    using typename GNAObject::SingleOutput;
    using typename GNAObject::OutputDescriptor;
    using typename GNAObject::InputDescriptor;
    using TransformationDescriptor = typename GNAObject::TransformationDescriptorType;

    GNAObjectBind1N(const std::string& transformation, const std::string& input, const std::string& output,
                    size_t transformation_offsset, size_t input_offset, size_t output_offset);

    virtual TransformationDescriptor add_transformation(const std::string& name="");
    InputDescriptor  add_input(const std::string& iname="", const std::string& oname="");
    OutputDescriptor add_input(SingleOutput& output, const std::string& iname="", const std::string& oname="");
    OutputDescriptor add_inputs(SingleOutputsContainer& outputs, const std::string& oname="") { return add_input(*outputs[0], "", oname); }

protected:
    void set_transformation_offset(size_t offset) { m_transformation_offset=offset; }
    void set_input_offset(size_t offset)          { m_input_offset=offset; }
    void set_output_offset(size_t offset)         { m_output_offset=offset; }

    std::string new_transformation_name(const std::string& name);

    void bind_tfirst_tlast(size_t noutput, size_t ninput);
    void set_open_input()   { m_open_input=true; }
    void reset_open_input() { m_open_input=false; }

private:
    std::string new_name(const std::string& base, size_t num, size_t offset, const std::string& altname="");

    std::string m_transformation_name="";
    std::string m_input_name="";
    std::string m_output_name="";

    size_t m_transformation_offset=0u;
    size_t m_input_offset=0u;
    size_t m_output_offset=0u;

    bool m_open_input=false;
};

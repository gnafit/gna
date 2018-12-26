#include "GNAObjectBind1N.hh"

GNAObjectBind1N::GNAObjectBind1N(const std::string& transformation, const std::string& input, const std::string& output,
								 size_t transformation_offset, size_t input_offset, size_t output_offset) :
m_transformation_name(transformation), m_input_name(input), m_output_name(output),
m_transformation_offset(transformation_offset), m_input_offset(input_offset), m_output_offset(output_offset)
{
    /* code */
}

std::string GNAObjectBind1N::new_transformation_name(const std::string& name){
	return new_name(m_transformation_name, transformations.size(), m_transformation_offset, name);
}

TransformationDescriptor GNAObjectBind1N::add_transformation(const std::string& name){
	throw std::runtime_error(fmt::format("Unimplemented method add_transformation ({0}, {1})", m_transformation_name, name));
}

std::string GNAObjectBind1N::new_name(const std::string& base, size_t num, size_t offset, const std::string& altname){
	if(altname.size()>0u){
		return altname;
	}

	int real_num = num-offset;
	if(num) {
        fmt::format("{0}_{1:02d}", base, real_num);
	}
	return base;
}

InputDescriptor GNAObjectBind1N::add_input(const std::string& iname, const std::string& oname){
    auto trans=transformations.back();
    auto input=trans.inputs.back();

    if(input.bound()){
        input=trans.input( new_name(m_input_name, trans.inputs.size(), m_input_offset, iname) );
        trans.output( new_name(m_output_name, trans.outputs.size(), m_output_offset, oname) );
    }

    return input;
}

OutputDescriptor GNAObjectBind1N::add_input(SingleOutput& output, const std::string& iname, const std::string& oname){
    auto input=add_input(iname, oname);
	output.single() >> input;
    return OutputDescriptor(transformations.back().outputs.back());
}

void GNAObjectBind1N::bind_tfirst_tlast(size_t noutput, size_t ninput){
	auto trans0 = transformations.front();
	const auto& output = trans0.outputs[noutput];

	auto trans1 = transformations.back();
	const auto& input = trans1.inputs[ninput];

	output >> input;
}

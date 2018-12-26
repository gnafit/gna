#pragma once

#include "GNAObject.hh"

class GNAObjectBind1N: public GNAObject {
public:
	GNAObjectBind1N(const std::string& transformation, const std::string& input, const std::string& output,
					size_t transformation_offsset, size_t input_offset, size_t output_offset);

	virtual TransformationDescriptor add_transformation(const std::string& name="");
	InputDescriptor  add_input(const std::string& iname="", const std::string& oname="");
	OutputDescriptor add_input(SingleOutput& output, const std::string& iname="", const std::string& oname="");

protected:
	void set_transformation_offset(size_t offset) { m_transformation_offset=offset; }
	void set_input_offset(size_t offset)          { m_input_offset=offset; }
	void set_output_offset(size_t offset)         { m_output_offset=offset; }

	std::string new_transformation_name(const std::string& name);

	void bind_tfirst_tlast(size_t noutput, size_t ninput);

private:
	std::string new_name(const std::string& base, size_t num, size_t offset, const std::string& altname="");

	std::string m_transformation_name="";
	std::string m_input_name="";
	std::string m_output_name="";

	size_t m_transformation_offset=0u;
	size_t m_input_offset=0u;
	size_t m_output_offset=0u;
};

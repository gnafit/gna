#include "GNAObjectBindkN.hh"

GNAObjectBindkN::GNAObjectBindkN(const std::string& transformation, const std::vector<std::string>& inputs,
                const std::string& output,
                size_t transformation_offset, size_t input_offset, size_t output_offset) :
m_transformation_name(transformation), m_input_names(inputs), m_output_name(output),
m_transformation_offset(transformation_offset), m_input_offset(input_offset), m_output_offset(output_offset)
{
  /* code */
}

std::string GNAObjectBindkN::new_transformation_name(const std::string& name){
  auto newname=new_name(m_transformation_name, transformations.size(), m_transformation_offset, name);
  if(transformations.contains(newname)){
    throw std::runtime_error(fmt::format("Unable to add transformation {}. Already in the list.", newname));
  }
  return newname;
}

TransformationDescriptor GNAObjectBindkN::add_transformation(const std::string& name){
  throw std::runtime_error(fmt::format("Unimplemented method add_transformation ({0}, {1})", m_transformation_name, name));
}

std::string GNAObjectBindkN::new_name(const std::string& base, size_t num, size_t offset, const std::string& altname, size_t divisor){
  //printf("%s %zu %zu %s\n", base.c_st(), num, offset, altname.c_str());
  if(altname.size()>0u){
    return altname;
  }

  int real_num = (num-offset)/divisor;
  if(real_num) {
    return fmt::format("{0}_{1:02d}", base, real_num+1);
  }
  return base;
}

OutputDescriptor GNAObjectBindkN::add_inputs(const std::vector<std::string>& inames_arg, const std::string& oname){
  add_inputs(inames_arg);
  return add_output(oname);
}

OutputDescriptor GNAObjectBindkN::add_inputs(const SingleOutputsContainer& outputs, const std::string& oname){
  add_inputs(outputs);
  return add_output(oname);
}

OutputDescriptor GNAObjectBindkN::add_output(const std::string& oname){
  auto trans=transformations.back();
  auto newname=new_name(m_output_name, trans.outputs.size(), m_output_offset, oname);
  if(trans.outputs.contains(newname)){
      throw std::runtime_error(fmt::format("Unable to add output {}. Already in the list.", newname));
  }
  return trans.output(newname);
}

void GNAObjectBindkN::add_inputs_only(const std::vector<std::string>& inames_arg){
  const std::vector<std::string>* inames;
  if(!inames_arg.empty()){
    if(inames_arg.size()!=m_input_names.size()){
      throw std::runtime_error("Invalid number of input names specified");
    }
    inames = &inames_arg;
  }
  else{
    inames = &m_input_names;
  }

  for(auto& iname: *inames){
    add_input(iname);
  }
}

void GNAObjectBindkN::add_inputs_only(const SingleOutputsContainer& outputs){
  if(outputs.size()!=m_input_names.size()){
    throw std::runtime_error("Invalid number of input names specified");
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    add_input(*outputs[i], m_input_names[i]);
  }
}

InputDescriptor GNAObjectBindkN::add_input(const std::string& iname){
    auto trans=transformations.back();

    if(m_open_inputs){
        auto input=trans.inputs[-m_open_inputs];
        if(!input.bound()){
            --m_open_inputs;
            return input;
        }
        else{
          throw std::runtime_error("Unable to fine proper open input");
        }
    }

    auto newname=new_name(iname, trans.inputs.size(), m_input_offset, "", m_input_names.size());
    if(trans.inputs.contains(newname)){
        throw std::runtime_error(fmt::format("Unable to add input {}. Already in the list.", newname));
    }
    auto input=trans.input(newname);

    return input;
}

void GNAObjectBindkN::add_input(SingleOutput& output, const std::string& iname){
  auto input=add_input(iname);
  output.single() >> input;
}

void GNAObjectBindkN::bind_tfirst_tlast(size_t noutput, size_t ninput){
  auto trans0 = transformations.front();
  const auto& output = trans0.outputs[noutput];

  auto trans1 = transformations.back();
  const auto& input = trans1.inputs[ninput];

  output >> input;
}

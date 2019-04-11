#include "GNAObjectBind1N.hh"

template<typename FloatType>
GNAObjectBind1N<FloatType>::GNAObjectBind1N(const std::string& transformation, const std::string& input, const std::string& output,
                                 size_t transformation_offset, size_t input_offset, size_t output_offset) :
m_transformation_name(transformation), m_input_name(input), m_output_name(output),
m_transformation_offset(transformation_offset), m_input_offset(input_offset), m_output_offset(output_offset)
{
    /* code */
}

template<typename FloatType>
std::string GNAObjectBind1N<FloatType>::new_transformation_name(const std::string& name){
    auto newname=new_name(m_transformation_name, transformations.size(), m_transformation_offset, name);
    if(transformations.contains(newname)){
        throw std::runtime_error(fmt::format("Unable to add transformation {}. Already in the list.", newname));
    }
    return newname;
}
template<typename FloatType>
typename GNAObjectBind1N<FloatType>::TransformationDescriptor GNAObjectBind1N<FloatType>::add_transformation(const std::string& name){
    throw std::runtime_error(fmt::format("Unimplemented method add_transformation ({0}, {1})", m_transformation_name, name));
}

template<typename FloatType>
std::string GNAObjectBind1N<FloatType>::new_name(const std::string& base, size_t num, size_t offset, const std::string& altname){
    //printf("%s %zu %zu %s\n", base.c_st(), num, offset, altname.c_str());
    if(altname.size()>0u){
        return altname;
    }

    int real_num = num-offset;
    if(real_num) {
        return fmt::format("{0}_{1:02d}", base, real_num+1);
    }
    return base;
}

template<typename FloatType>
typename GNAObjectBind1N<FloatType>::InputDescriptor GNAObjectBind1N<FloatType>::add_input(const std::string& iname, const std::string& oname){
    auto trans=transformations.back();

    if(m_open_input){
        m_open_input = false;
        auto input=trans.inputs.back();
        if(!input.bound()){
            return input;
        }
    }

    auto newname=new_name(m_input_name, trans.inputs.size(), m_input_offset, iname);
    if(trans.inputs.contains(newname)){
        throw std::runtime_error(fmt::format("Unable to add input {}. Already in the list.", newname));
    }
    auto input=trans.input(newname);

    newname=new_name(m_output_name, trans.outputs.size(), m_output_offset, oname);
    if(trans.outputs.contains(newname)){
        throw std::runtime_error(fmt::format("Unable to add output {}. Already in the list.", newname));
    }
    trans.output(newname);

    return input;
}

template<typename FloatType>
typename GNAObjectBind1N<FloatType>::OutputDescriptor GNAObjectBind1N<FloatType>::add_input(typename GNAObjectBind1N<FloatType>::SingleOutput& output, const std::string& iname, const std::string& oname){
    auto input=add_input(iname, oname);
    output.single() >> input;
    return OutputDescriptor(transformations.back().outputs.back());
}

template<typename FloatType>
void GNAObjectBind1N<FloatType>::bind_tfirst_tlast(size_t noutput, size_t ninput){
    auto trans0 = transformations.front();
    const auto& output = trans0.outputs[noutput];

    auto trans1 = transformations.back();
    const auto& input = trans1.inputs[ninput];

    output >> input;
}

template class GNAObjectBind1N<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNAObjectBind1N<float>;
#endif

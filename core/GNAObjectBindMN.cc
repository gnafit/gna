#include "GNAObjectBindMN.hh"

template<typename FloatType>
GNAObjectBindMN<FloatType>::GNAObjectBindMN(const std::string& transformation, const std::string& input, const std::string& output) :
m_transformation_name(transformation), m_input_name(input), m_output_name(output)
{
    /* code */
}

template<typename FloatType>
std::string GNAObjectBindMN<FloatType>::new_transformation_name(const std::string& name){
    auto newname=new_name(m_transformation_name, transformations.size(), name);
    if(transformations.contains(newname)){
        throw std::runtime_error(fmt::format("Unable to add transformation {}. Already in the list.", newname));
    }
    return newname;
}

template<typename FloatType>
TransformationDescriptor GNAObjectBindMN<FloatType>::add_transformation(const std::string& name){
    throw std::runtime_error(fmt::format("Unimplemented method add_transformation ({0}, {1})", m_transformation_name, name));
}

template<typename FloatType>
std::string GNAObjectBindMN<FloatType>::new_name(const std::string& base, size_t num, const std::string& altname){
    //printf("%s %zu %zu %s\n", base.c_str(), num, altname.c_str());
    if(altname.size()>0u){
        return altname;
    }

    if(num) {
        return fmt::format("{0}_{1:02d}", base, num+1);
    }
    return base;
}

template<typename FloatType>
typename GNAObjectBindMN<FloatType>::OutputDescriptor GNAObjectBindMN<FloatType>::add_output(const std::string& oname){
    auto trans=transformations.back();
    auto newname=new_name(m_output_name, trans.outputs.size(), oname);
    if(trans.outputs.contains(newname)){
        throw std::runtime_error(fmt::format("Unable to add output {}. Already in the list.", newname));
    }
    return trans.output(newname);
}

template<typename FloatType>
typename GNAObjectBindMN<FloatType>::InputDescriptor GNAObjectBindMN<FloatType>::add_input(const std::string& iname){
    auto trans=transformations.back();

    if(m_open_input){
        m_open_input = false;
        auto input=trans.inputs.back();
        if(!input.bound()){
            return input;
        }
    }

    auto newname=new_name(m_input_name, trans.inputs.size(), iname);
    if(trans.inputs.contains(newname)){
        throw std::runtime_error(fmt::format("Unable to add input {}. Already in the list.", newname));
    }
    auto input=trans.input(newname, -1);

    return input;
}

template<typename FloatType>
typename GNAObjectBindMN<FloatType>::OutputDescriptor GNAObjectBindMN<FloatType>::add_input(typename GNAObjectBindMN<FloatType>::SingleOutput& output, const std::string& iname){
    auto input=add_input(iname);
    output.single() >> input;
    return OutputDescriptor(transformations.back().outputs.back());
}

template<typename FloatType>
void GNAObjectBindMN<FloatType>::add_inputs(const typename GNAObjectBindMN<FloatType>::OutputDescriptors& outputs){
    for(const auto& output: outputs){
        add_input(*output);
    }
}

template class GNAObjectBindMN<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNAObjectBindMN<float>;
#endif

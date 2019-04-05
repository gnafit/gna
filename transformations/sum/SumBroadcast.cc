#include "SumBroadcast.hh"
#include "TypesFunctions.hh"

SumBroadcast::SumBroadcast() {
  transformation_("sum")
    .output("sum")
    .types(TypesFunctions::ifSameShapeOrSingle, TypesFunctions::passNonSingle<0,0>)
    .func([](FunctionArgs& fargs) {
        auto& args=fargs.args;
        auto& ret=fargs.rets[0].x;
        double offset=0.0;
        bool secondary=false;
        for (size_t i = 0; i < args.size(); ++i) {
          auto& data=args[i].x;
          if (data.size()!=1) {
            if (secondary) {
              ret+=data;
            } else {
              ret=data;
              secondary=true;
            }
          }
          else{
            offset+=data(0);
          }
        }
        if(!secondary){
          ret=offset;
        }
        else if(offset!=0.0){
          ret+=offset;
        }
      });
}

/**
 * @brief Construct SumBroadcast from vector of SingleOutput instances
 */
SumBroadcast::SumBroadcast(const OutputDescriptor::OutputDescriptors& outputs) : SumBroadcast(){
  for(auto& output : outputs){
    this->add(*output);
  }
}

/**
 * @brief Add an input and connect it to the output.
 *
 * The input name is derived from the output name.
 *
 * @param out -- a SingleOutput instance.
 * @return InputDescriptor instance for the newly created input.
 */
InputDescriptor SumBroadcast::add(SingleOutput &out) {
  return InputDescriptor(t_[0].input(out));
}

/**
 * @brief Add an input by name and leave unconnected.
 * @param name -- a name for the new input.
 * @return InputDescriptor instance for the newly created input.
 */
InputDescriptor SumBroadcast::add_input(const char* name) {
  return InputDescriptor(t_[0].input(name));
}

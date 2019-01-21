#include "Product.hh"
#include "TypesFunctions.hh"

Product::Product() {
  transformation_("product")
    .output("product")
    .types(TypesFunctions::ifSameShapeOrSingle, TypesFunctions::passNonSingle<0,0>)
    .func([](FunctionArgs& fargs) {
        auto& args=fargs.args;
        auto& ret=fargs.rets[0].x;
        double factor=1.0;
        bool secondary=false;
        for (size_t i = 0; i < args.size(); ++i) {
          auto& data=args[i].x;
          if (data.size()!=1) {
            if (secondary) {
              ret*=data;
            } else {
              ret=data;
              secondary=true;
            }
          }
          else{
            factor*=data(0);
          }
        }
        if(!secondary){
          ret=factor;
        }
        else if(factor!=1){
          ret*=factor;
        }
      });
}

/**
 * @brief Construct Product from vector of SingleOutput instances
 */
Product::Product(const OutputDescriptor::OutputDescriptors& outputs) : Product(){
  for(auto& output : outputs){
    this->multiply(*output);
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
InputDescriptor Product::multiply(SingleOutput &out) {
  return InputDescriptor(t_[0].input(out));
}

/**
 * @brief Add an input by name and leave unconnected.
 * @param name -- a name for the new input.
 * @return InputDescriptor instance for the newly created input.
 */
InputDescriptor Product::add_input(const char* name) {
  return InputDescriptor(t_[0].input(name));
}

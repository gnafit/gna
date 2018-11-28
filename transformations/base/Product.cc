#include "Product.hh"

Product::Product() {
  transformation_("product")
    .output("product")
    .types([](TypesFunctionArgs& fargs) {
        auto& args=fargs.args;
        auto& rets=fargs.rets;
        DataType dtsingle = DataType().points().shape(1);
        DataType dt = dtsingle;
        for (size_t i = 0; i < args.size(); ++i) {
          if (args[i] == dtsingle) {
            continue;
          }
          if (dt != dtsingle && args[i] != dt) {
            throw rets.error(rets[0]);
          }
          dt = args[i];
        }
        rets[0] = dt;
      })
    .func([](FunctionArgs& fargs) {
        auto& args=fargs.args;
        auto& rettype=fargs.rets[0].type;
        auto& ret=fargs.rets[0].x;
        size_t i;
        for (i = 0; i < args.size(); ++i) {
          if (args[i].type == rettype) {
            ret = args[i].x;
            break;
          }
        }
        for (size_t j = 0; j < args.size(); ++j) {
          if (args[j].x.size() == 1) {
            ret *= args[j].x(0);
          } else if (j != i) {
            ret *= args[j].x;
          }
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

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

void Product::multiply(SingleOutput &out) {
  t_["product"].input(out);
}

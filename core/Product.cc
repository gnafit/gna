#include "Product.hh"

Product::Product() {
  transformation_(this, "product")
    .output("product", DataType().points().any())
    .types([](Atypes args, Rtypes rets) {
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
    .func([](Args args, Rets rets) {
        size_t i;
        for (i = 0; i < args.size(); ++i) {
          if (args[i].type == rets[0].type) {
            rets[0].x = args[i].x;
            break;
          }
        }
        for (size_t j = 0; j < args.size(); ++j) {
          if (args[j].x.size() == 1) {
            rets[0].x *= args[j].x(0);
          } else if (j != i) {
            rets[0].x *= args[j].x;
          }
        }
      });
}

void Product::multiply(const OutputDescriptor &out) {
  t_["product"].input(out.channel()).connect(out);
}

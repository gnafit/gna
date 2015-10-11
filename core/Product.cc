#include "Product.hh"

Product::Product() {
  transformation_(this, "product")
    .output("product", DataType().points().any())
    .types(Atypes::ifSame, Atypes::pass<0>)
    .func([](Args args, Rets rets) {
        rets[0].x = args[0].x;
        for (size_t i = 1; i < args.size(); ++i) {
          rets[0].x *= args[i].x;
        }
      });
}

void Product::multiply(const OutputDescriptor &out) {
  t_["product"].input(out.channel()).connect(out);
}

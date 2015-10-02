#include "Product.hh"

Product::Product() {
  transformation_("product")
    .output("product", DataType().points().any())
    .types([](Atypes args, Rtypes rets) {
        if (args.size() == 0) {
          return Status::Undefined;
        }
        for (size_t i = 1; i < args.size(); ++i) {
          if (args[i] != args[0]) {
            return Status::Failed;
          }
        }
        rets[0] = args[0];
        return Status::Success;
      })
    .func([](Args args, Rets rets) {
        rets[0].x = args[0].x;
        for (size_t i = 1; i < args.size(); ++i) {
          rets[0].x *= args[i].x;
        }
        return Status::Success;
      });
}

void Product::add(const OutputDescriptor &out) {
  t_["product"].input(out.channel()).connect(out);
}

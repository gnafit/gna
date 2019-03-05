#include "Sum.hh"

Sum::Sum() {
  transformation_(this, "sum")
    .output("sum")
    .types(Atypes::ifSame, Atypes::pass<0>)
    .func([](Args args, Rets rets) {
        rets[0].x = args[0].x;
        for (size_t j = 1; j < args.size(); ++j) {
          rets[0].x += args[j].x;
        }
      });
}

void Sum::add(SingleOutput &out) {
  t_["sum"].input(out);
}

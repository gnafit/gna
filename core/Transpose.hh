#include "GNAObject.hh"
#include <algorithm>

class Transpose: public GNAObject,
                 public Transformation<Transpose> {
    public:
        Transpose() {
            transformation_(this, "transpose")
                .input("mat")
                .output("T")
                .types([](Transpose* obj, Atypes args, Rtypes rets){
                        auto reversed_shape = args[0].shape;
                        std::reverse(std::begin(reversed_shape), std::end(reversed_shape));
                        rets[0] = DataType().points().shape(reversed_shape);
                        })
                .func([](Transpose* obj, Args args, Rets rets){
                        rets[0].mat = args[0].mat.transpose();
                        });
        }
                         
};

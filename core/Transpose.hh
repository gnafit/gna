#include "GNAObject.hh"
#include <algorithm>
#include <iterator>

class Transpose: public GNAObject,
                 public Transformation<Transpose> {
    public:
        Transpose() {
            transformation_(this, "transpose")
                .input("mat")
                .output("T")
                .types([](Transpose* obj, Atypes args, Rtypes rets){
                        auto shape = args[0].shape;
                        rets[0] = DataType().points().shape({shape.rbegin(), shape.rend()});
                        })
                .func([](Transpose* obj, Args args, Rets rets){
                        rets[0].mat = args[0].mat.transpose();
                        });
        }
                         
};

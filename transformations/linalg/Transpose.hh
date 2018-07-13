#include "GNAObject.hh"

class Transpose: public GNAObject,
                 public TransformationBind<Transpose> {
    public:
        Transpose() {
            transformation_("transpose")
                .input("mat")
                .output("T")
                .types([](Transpose* obj, Atypes args, Rtypes rets){
                        auto input_shape = args[0].shape;
                        rets[0] = DataType().points().shape({input_shape.rbegin(), input_shape.rend()});
                        })
                .func([](Transpose* obj, FunctionArgs fargs){
                        fargs.rets[0].mat = fargs.args[0].mat.transpose();
                        });
        }

};

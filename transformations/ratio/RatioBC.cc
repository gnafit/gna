#include "RatioBC.hh"
#include "TypesFunctions.hh"

void compute_ratio(TransformationTypes::FunctionArgsT<double,double>& fargs){
    auto& args=fargs.args;
    auto& shape1=args[0].type.shape;
    auto& shape2=args[1].type.shape;

    if(shape1.size()==shape2.size()){
        fargs.rets[0].x = args[0].x/args[1].x;
        return;
    }
    auto& ret=fargs.rets[0].arr2d;
    if(shape1.size()<shape2.size()){
        ret.colwise() = args[0].arr;
        ret/=args[1].arr2d;
        return;
    }
    ret = args[0].arr2d;
    ret.colwise()/=args[1].arr;

}

void check_types(TransformationTypes::TypesFunctionArgsT<double,double>& fargs){
    auto& args=fargs.args;
    auto& rets=fargs.rets;

    auto& shape1=args[0].shape;
    auto& shape2=args[1].shape;

    if(shape1==shape2){
        rets[0]=args[0];
        return;
    }
    assert(shape1.size()!=shape2.size());
    if(shape2.size()<shape1.size()){
        rets[0]=args[0];
    }
    else if(shape2.size()>shape1.size()){
        rets[0]=args[1];
    }
    assert(shape1[0]==shape2[0]);
}

/**
 * @brief Constructor.
 */
RatioBC::RatioBC() {
    transformation_("ratio")
        .input("top")
        .input("bottom")
        .output("ratio")
        .types(&check_types)
        .func(&compute_ratio);
}

/**
 * @brief Construct ratio of top and bottom
 * @param top — nominator
 * @param bottom — denominator
 */
RatioBC::RatioBC(SingleOutput& top, SingleOutput& bottom) : RatioBC() {
    divide(top, bottom);
}


/**
 * @brief Bind nomenator, denomenator and return the ratio (output)
 * @param top — nominator
 * @param bottom — denominator
 * @return the ratio output
 */
OutputDescriptor RatioBC::divide(SingleOutput& top, SingleOutput& bottom){
    const auto& t = t_[0];
    const auto& inputs = t.inputs();
    inputs[0].connect(top.single());
    inputs[1].connect(bottom.single());
    return OutputDescriptor(t.outputs()[0]);
}


#include "SumMatOrDiag.hh"

SumMatOrDiag::SumMatOrDiag() {
    transformation_("sum")
        .output("sum")
        .types(&SumMatOrDiag::checkTypes)
        .func(&SumMatOrDiag::calculateSum);
}

/**
 * @brief Construct SumMatOrDiag from vector of SingleOutput instances
 */
SumMatOrDiag::SumMatOrDiag(const OutputDescriptor::OutputDescriptors& outputs) : SumMatOrDiag() {
    for (auto output : outputs) {
        this->add(output);
    }
}

/**
 * @brief Add an input.
 *
 * @param data -- a SingleOutput instance.
 * @return InputDescriptor instance for the newly created input.
 */
InputDescriptor SumMatOrDiag::add(SingleOutput& data) const {
    return InputDescriptor(t_[0].input(data));
}

/**
 * @brief Check that the input matrices are square and all the inputs have the same dimension.
 */
void SumMatOrDiag::checkTypes(TypesFunctionArgs fargs) const {
    auto& args = fargs.args;
    auto shape = &args[0].shape;
    for (size_t i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        switch (arg.shape.size()) {
            case 1:
                if ((*shape)[0] != arg.shape[0])
                    throw args.error(arg, fmt::format("Arg {0} has incorrect shape: {1} != {2}", i,
                                                      arg.shape[0], (*shape)[0]));
                break;
            case 2:
                if ((*shape)[0] != arg.shape[0] || (*shape)[0] != arg.shape[1])
                    throw args.error(
                        arg, fmt::format("Arg {0} has incorrect shape: "
                                         "({1}, {2}) != ({3}, {4})",
                                         i, arg.shape[0], arg.shape[1], (*shape)[0], (*shape)[0]));
                if ((*shape).size() < arg.shape.size()) shape = &arg.shape;
                break;
            default:
                throw args.error(arg, fmt::format("Arg {0} has a dimension of more than 2: {1} > 2",
                                                  i, arg.shape.size()));
                break;
        }
    }
    fargs.rets[0] = DataType().points().shape(*shape);
}

/**
 * @brief Calculate matrix sum of the inputs.
 */
void SumMatOrDiag::calculateSum(FunctionArgs& fargs) const {
    size_t ndim = fargs.rets[0].type.shape.size();
    switch (ndim) {
        case 1:
            sumVec(fargs);
            break;
        case 2:
            sumMat(fargs);
            break;
        default:
            throw fargs.rets.error(fmt::format("Ret has dimension of more than 2: {0} > 2", ndim));
            break;
    }
}

/**
 * @brief Sum the inputs as vectors.
 */
void SumMatOrDiag::sumVec(FunctionArgs& fargs) const {
    const auto& args = fargs.args;
    auto& ret = fargs.rets[0].x;
    ret = args[0].x;
    for (size_t i = 1; i < args.size(); ++i) {
        ret += args[i].x;
    }
}

/**
 * @brief Sum the inputs as matrices.
 */
void SumMatOrDiag::sumMat(FunctionArgs& fargs) const {
    const auto& args = fargs.args;
    auto& ret = fargs.rets[0].mat.setZero();
    for (size_t i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        switch (arg.type.shape.size()) {
            case 1:
                ret.diagonal() += arg.vec;
                break;
            case 2:
                ret += arg.mat;
                break;
            default:
                break;
        }
    }
}

#include "Jacobian.hh"
#include "EigenHelpers.hh"
#include <algorithm>

void Jacobian::calcJacobian(FunctionArgs fargs) {
    auto& args = fargs.args;
    auto& arg  = args[0].x;
    auto& ret  = fargs.rets[0].mat;
    ret.setZero();

    Eigen::ArrayXd ret1(arg.size());
    std::array<double, 4> points;
    for (size_t i=0; i < m_pars.size(); ++i) {
        auto* x = m_pars.at(i);
        auto x0 = x->value();
        auto reldelta_corrected = m_reldelta*x->step();

        double f1 = 4.0/(3.0*reldelta_corrected);
        double f2 = 1.0/(6.0*reldelta_corrected);

        points[0] = x->relativeValue(+m_reldelta/2);
        points[1] = x->relativeValue(-m_reldelta/2);
        points[2] = x->relativeValue(+m_reldelta);
        points[3] = x->relativeValue(-m_reldelta);

        x->set(points[0]); args.touch(); ret1  = f1*arg;
        x->set(points[1]); args.touch(); ret1 -= f1*arg;
        x->set(points[2]); args.touch(); ret1 -= f2*arg;
        x->set(points[3]); args.touch(); ret1 += f2*arg;
        x->set(x0);        args.touch();

        ret.col(i) = ret1.matrix();
    }

    fargs.rets.untaint();
    fargs.rets.freeze();
}

void Jacobian::calcTypes(TypesFunctionArgs fargs){
    fargs.rets[0] = DataType().points().shape(fargs.args[0].size(), m_pars.size());
}

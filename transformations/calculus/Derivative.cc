#include "Derivative.hh"

void Derivative::calcDerivative(FunctionArgs fargs) {
    auto& args = fargs.args;
    auto& arg  = args[0].x;
    auto& ret  = fargs.rets[0].x;
    auto x0 = m_x->value();
    auto reldelta_corrected = m_reldelta*m_x->step();

    double f1 = 4.0/(3.0*reldelta_corrected);
    double f2 = 1.0/(6.0*reldelta_corrected);

    std::array<double, 4> points;
    points[0] = m_x->relativeValue(+m_reldelta/2);
    points[1] = m_x->relativeValue(-m_reldelta/2);
    points[2] = m_x->relativeValue(+m_reldelta);
    points[3] = m_x->relativeValue(-m_reldelta);

    m_x->set(points[0]); args.touch(); ret  = f1*arg;
    m_x->set(points[1]); args.touch(); ret -= f1*arg;
    m_x->set(points[2]); args.touch(); ret -= f2*arg;
    m_x->set(points[3]); args.touch(); ret += f2*arg;
    m_x->set(x0);        args.touch();

    fargs.rets.untaint();
    fargs.rets.freeze();
}

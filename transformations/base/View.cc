#include "View.hh"
#include "TypeClasses.hh"

View::View(size_t start) : m_start(start) {
    init();
}

View::View(size_t start, size_t len) : m_start(start), m_len(len) {
    init();
}

View::View(SingleOutput* output, size_t start, size_t len) :
View(start, len)
{
    output->single() >> this->transformations.back().inputs.back();
}

View::View(SingleOutput* output, size_t start) :
View(start)
{
    output->single() >> this->transformations.back().inputs.back();
}

void View::init(){
    this->transformation_("view")
         .input("data")
         .output("view")
         .types(&View::types)
         .func([](FunctionArgs& fargs){fargs.args.touch();});

}

void View::types(TypesFunctionArgs& fargs){
    auto& args = fargs.args;
    auto& rets = fargs.rets;
    for (size_t i = 0; i < args.size(); ++i) {
        auto& arg = args[i];

        bool truncated = m_start || m_len.has_value();
        if(truncated && arg.shape.size()>1){
            throw args.error(arg, fmt::format("Transformation {0}: input {1} dim should be 1", args.name(), i));
        }

        // Check that the limits are sane
        auto input_size = arg.size();
        if(m_start>=input_size)
        {
            throw args.error(arg, fmt::format("Transformation {0}: input {1} length {2} <= offset {3}",
                                              args.name(), i, input_size, m_start));
        }

        size_t len=static_cast<size_t>(input_size - m_start);
        if (m_len.has_value()){
            if(*m_len>len)
            {
                throw args.error(arg, fmt::format("Transformation {0}: input {1} length {2} is not sufficient for offset {3} and len {4}",
                                                  args.name(), i, input_size, m_start, *m_len));
            }

            len = *m_len;
        }

        double* buf = args.data(i).buffer+m_start;
        auto& ret=rets[i];
        switch(arg.kind){
            case DataKind::Hist:
                if (truncated){
                    ret.hist().edges(len+1, arg.edges.data()+m_start).preallocated(buf);
                }
                else{
                    ret=arg;
                    ret.hist().preallocated(buf);
                }
                break;
            case DataKind::Points:
                if(truncated){
                    ret.points().shape(len).preallocated(buf);
                }
                else{
                    ret=arg;
                    ret.points().preallocated(buf);
                }
                break;
            default:
                continue;
                break;
        }
    }
}


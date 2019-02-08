#include "Dummy.hh"
#include "cuElementary.hh"

using TransformationTypes::GPUShape;

/**
 * @brief Constructor.
 * @param shape - size of each output.
 * @param label - transformation label.
 * @param labels - variables to bind.
 */
Dummy::Dummy(size_t shape, const char* label, const std::vector<std::string> &labels){
  transformation_("dummy")
    .label(label)
    .types([shape](TypesFunctionArgs fargs) {
              for (size_t i = 0; i < fargs.rets.size(); ++i) {
                fargs.rets[i] = DataType().points().shape(shape);
              }
            }
          )
  .func(&Dummy::dummy_fcn)
  .func("dummy_gpuargs_h", &Dummy::dummy_gpuargs_h/*, DataLocation::Device*/);

  m_vars.resize(labels.size());
  for (size_t i = 0; i < m_vars.size(); ++i) {
    variable_(&m_vars[i], labels[i].data());
  }
}

void Dummy::dummy_fcn(FunctionArgs& fargs){
    auto& rets = fargs.rets;
    for (size_t i = 0; i < rets.size(); ++i) {
        rets[i].x=static_cast<double>(i);
    }
}

void Dummy::dummy_gpuargs_h(FunctionArgs& fargs){
    fargs.args.touch();
    auto& gpuargs=fargs.gpu;
    gpuargs->readVariables(m_vars);
    gpuargs->provideSignatureHost();

    for (size_t i = 0; i < gpuargs->nrets; ++i) {
        auto* shape =gpuargs->retshapes[i];
        auto size=shape[(int)GPUShape::Size];

        auto* start=*std::next(gpuargs->rets, i);
        auto* stop = std::next(start, size);
        std::fill(start, stop, static_cast<double>(i));
    }

    gpuargs->dump();
}


#include "Dummy.hh"
#include "config_vars.h"
#ifdef GNA_CUDA_SUPPORT
#include "cuElementary.hh"
#endif

using TransformationTypes::GPUShape;

/**
 * @brief Constructor.
 * @param shape - size of each output.
 * @param label - transformation label.
 * @param labels - variables to bind.
 */
template<typename FloatType>
GNA::GNAObjectTemplates::DummyT<FloatType>::DummyT(size_t shape, const char* label, const std::vector<std::string> &labels){
  this->transformation_("dummy")
    .label(label)
    .types([shape](typename GNAObjectT<FloatType,FloatType>::TypesFunctionArgs fargs) {
              for (size_t i = 0; i < fargs.rets.size(); ++i) {
                fargs.rets[i] = DataType().points().shape(shape);
              }
            }
          )
  .func(&DummyType::dummy_fcn)
#ifdef GNA_CUDA_SUPPORT
  .func("dummy_gpuargs_h_local", &DummyType::dummy_gpuargs_h_local/*, DataLocation::Host*/)
  .func("dummy_gpuargs_h",       &DummyType::dummy_gpuargs_h/*,       DataLocation::Host*/)
  .func("dummy_gpuargs_d",       &DummyType::dummy_gpuargs_d,       DataLocation::Device)
#endif
  .finalize();

  m_vars.resize(labels.size());
  for (size_t i = 0; i < m_vars.size(); ++i) {
    this->variable_(&m_vars[i], labels[i].data());
  }
}

template<typename FloatType>
void GNA::GNAObjectTemplates::DummyT<FloatType>::dummy_fcn(FunctionArgs& fargs){
    auto& rets = fargs.rets;
    for (size_t i = 0; i < rets.size(); ++i) {
        rets[i].x=static_cast<FloatType>(i);
    }
}

#ifdef GNA_CUDA_SUPPORT
template<typename FloatType>
void GNA::GNAObjectTemplates::DummyT<FloatType>::dummy_gpuargs_h_local(FunctionArgs& fargs){
    fargs.args.touch();
    auto& gpuargs=fargs.gpu;
    gpuargs->readVariablesLocal();
    gpuargs->provideSignatureHost(true /*local*/);

    for (size_t i = 0; i < gpuargs->nrets; ++i) {
        auto* shape =gpuargs->retshapes[i];
        auto size=shape[(int)GPUShape::Size];

        auto* start=*std::next(gpuargs->rets, i);
        auto* stop = std::next(start, size);
        std::fill(start, stop, static_cast<FloatType>(i));
    }

    gpuargs->dump();
}

template<typename FloatType>
void GNA::GNAObjectTemplates::DummyT<FloatType>::dummy_gpuargs_h(FunctionArgs& fargs){
    fargs.args.touch();
    auto& gpuargs=fargs.gpu;
    gpuargs->provideSignatureHost(); /*global*/

    for (size_t i = 0; i < gpuargs->nrets; ++i) {
        auto* shape =gpuargs->retshapes[i];
        auto size=shape[(int)GPUShape::Size];

        auto* start=*std::next(gpuargs->rets, i);
        auto* stop = std::next(start, size);
        std::fill(start, stop, static_cast<FloatType>(i));
    }

    gpuargs->dump();
}

template<typename FloatType>
void GNA::GNAObjectTemplates::DummyT<FloatType>::dummy_gpuargs_d(FunctionArgs& fargs){
    fargs.args.touch();
    auto& gpuargs=fargs.gpu;
    gpuargs->provideSignatureDevice(); /*global*/
    for (size_t i = 0; i < gpuargs->nrets; i++) {
        auto size = fargs.rets[i].arr.size();
	cufilllike(i, gpuargs->rets, static_cast<int> (size) );
    }
    std::cout << std::endl << "GPU dump:" <<std::endl;
    for (size_t i = 0; i < gpuargs->nrets; i++) {
	fargs.rets[i].gpuArr->dump();
    }
    std::cout << __PRETTY_FUNCTION__ << std::endl <<  "nvars = " << gpuargs->nvars << std::endl <<std::endl;
    debug_drop(gpuargs->vars, 3, 1);
     //gpuargs->dump();
}
#endif

template class GNA::GNAObjectTemplates::DummyT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::DummyT<float>;
#endif

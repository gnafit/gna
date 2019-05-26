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
  .func("dummy_gpuargs_h_local", &DummyType::dummy_gpuargs_h_local/*, DataLocation::Host*/)
  .func("dummy_gpuargs_h",       &DummyType::dummy_gpuargs_h/*,       DataLocation::Host*/)
  .func("dummy_gpuargs_d",       &DummyType::dummy_gpuargs_d,       DataLocation::Device)
  .finalize();

  m_vars.resize(labels.size());
  for (size_t i = 0; i < m_vars.size(); ++i) {
    this->variable_(&m_vars[i], labels[i].data());
  }
}

template<typename FloatType>
void GNA::GNAObjectTemplates::DummyT<FloatType>::dummy_fcn(typename GNAObjectT<FloatType,FloatType>::FunctionArgs& fargs){
    auto& rets = fargs.rets;
    for (size_t i = 0; i < rets.size(); ++i) {
        rets[i].x=static_cast<FloatType>(i);
    }
}

template<typename FloatType>
void GNA::GNAObjectTemplates::DummyT<FloatType>::dummy_gpuargs_h_local(typename GNAObjectT<FloatType,FloatType>::FunctionArgs& fargs){
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
void GNA::GNAObjectTemplates::DummyT<FloatType>::dummy_gpuargs_h(typename GNAObjectT<FloatType,FloatType>::FunctionArgs& fargs){
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
void GNA::GNAObjectTemplates::DummyT<FloatType>::dummy_gpuargs_d(typename GNAObjectT<FloatType,FloatType>::FunctionArgs& fargs){
#ifdef GNA_CUDA_SUPPORT
  
    fargs.args.touch();
    auto& gpuargs=fargs.gpu;
    gpuargs->provideSignatureDevice(); /*global*/
    for (size_t i = 0; i < gpuargs->nrets; i++) {
        auto size = fargs.rets[i].arr.size();
        
	std::cout  << "TMP i=" << i <<std::endl;
	cufilllike(i, gpuargs->rets, static_cast<int> (size) );
    }
    std::cout << std::endl << "GPU dump:" <<std::endl;
    for (size_t i = 0; i < gpuargs->nrets; i++) {
	fargs.rets[i].gpuArr->dump();
    }
     //gpuargs->dump();

#else
  throw std::runtime_error("CUDA support not implemented");
#endif
}

template class GNA::GNAObjectTemplates::DummyT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class GNA::GNAObjectTemplates::DummyT<float>;
#endif

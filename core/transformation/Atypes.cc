#include "Atypes.hh"

#include "TransformationEntry.hh"

using TransformationTypes::AtypesT;
using TransformationTypes::SourceT;
using TransformationTypes::SourceTypeError;

/**
 * @brief Source type exception.
 * @param dt -- incorrect DataType.
 * @param message -- exception message.
 * @return exception.
 */
template<typename SourceFloatType, typename SinkFloatType>
SourceTypeError<SourceT<SourceFloatType>> AtypesT<SourceFloatType,SinkFloatType>::error(const DataType &dt, const std::string &message) {
  const SourceImpl *source = nullptr;
  for (size_t i = 0; i < m_entry->sources.size(); ++i) {
    auto& lsource=m_entry->sources[i];
    if (&lsource.sink->data->type == &dt) {
      source = &lsource;
      break;
    }
  }
  return SourceTypeError<SourceImpl>(source, message);
}

template class TransformationTypes::AtypesT<double,double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::AtypesT<float,float>;
#endif

#include "InputHandle.hh"
#include "OutputHandle.hh"

/**
 * @brief Connect the Source to the other transformation's Sink via its OutputHandle
 * @param out -- OutputHandle view to the Sink to connect to.
 */
template<typename FloatType>
void TransformationTypes::InputHandleT<FloatType>::connect(const TransformationTypes::OutputHandleT<FloatType> &out) const {
  return m_source->connect(out.m_sink);
}

template class TransformationTypes::InputHandleT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::InputHandleT<float>;
#endif

#include "InputHandle.hh"
#include "OutputHandle.hh"

using TransformationTypes::InputHandle;
using TransformationTypes::OutputHandle;

/**
 * @brief Connect the Source to the other transformation's Sink via its OutputHandle
 * @param out -- OutputHandle view to the Sink to connect to.
 */
void InputHandle::connect(const OutputHandle &out) const {
  return m_source->connect(out.m_sink);
}

#include "Atypes.hh"

using TransformationTypes::Atypes;
using TransformationTypes::Rtypes;
using TransformationTypes::SourceTypeError;

/**
 * @brief Source type exception.
 * @param dt -- incorrect DataType.
 * @param message -- exception message.
 * @return exception.
 */
SourceTypeError Atypes::error(const DataType &dt, const std::string &message) {
  const Source *source = nullptr;
  for (size_t i = 0; i < m_entry->sources.size(); ++i) {
    if (&m_entry->sources[i].sink->data->type == &dt) {
      source = &m_entry->sources[i];
      break;
    }
  }
  return SourceTypeError(source, message);
}


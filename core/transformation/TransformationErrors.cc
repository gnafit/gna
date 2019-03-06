#include "TransformationErrors.hh"

#include "Sink.hh"
#include "Source.hh"
#include "TransformationEntry.hh"

#include <fmt/format.h>

using TransformationTypes::SinkTypeError;
using TransformationTypes::SourceTypeError;
using TransformationTypes::StorageTypeError;
using TransformationTypes::CalculationError;

/**
 * @brief Formatted exception message.
 * @tparam T -- type of the class to get the ->name from.
 * @param type -- exception's source.
 * @param s -- the ->name of this class will be shown in a message.
 * @param msg -- the actual message.
 */
template <typename T>
std::string errorMessage(const std::string &type, const T *s,
                         const std::string &msg) {
  std::string name;
  if (s) {
    name = std::string(" ")+s->name;
  }
  std::string message = msg.empty() ? "unspecified error" : msg;
  return fmt::format("{1}{0}: {2}", name, type, message);
}

/** @brief Constructor.
 *  @param s -- Sink with problematic type.
 *  @param message -- error message.
 */
template<typename SinkType>
SinkTypeError<SinkType>::SinkTypeError(const SinkType *s, const std::string &message)
  : TypeError(errorMessage("sink", s, message)),
    sink(s)
{ }

/** @brief Constructor.
 *  @param s -- Storage with problematic type.
 *  @param message -- error message.
 */
template<typename StorageType>
StorageTypeError<StorageType>::StorageTypeError(const StorageType *s, const std::string &message)
  : TypeError(errorMessage("storage", s, message)),
    storage(s)
{ }

/** @brief Constructor.
 *  @param s -- Source with problematic type.
 *  @param message -- error message.
 */
template<typename SourceType>
SourceTypeError<SourceType>::SourceTypeError(const SourceType *s, const std::string &message)
  : TypeError(errorMessage("source", s, message)),
    source(s)
{ }

/** @brief Constructor.
 *  @param e -- Entry where exception happens.
 *  @param message -- error message.
 */
template<typename EntryType>
CalculationError<EntryType>::CalculationError(const EntryType *e, const std::string &message)
  : std::runtime_error(errorMessage("transformation", e, message)),
    entry(e)
{ }

template class TransformationTypes::CalculationError<TransformationTypes::EntryT<double,double>>;
template class TransformationTypes::SourceTypeError<TransformationTypes::SourceT<double>>;
template class TransformationTypes::SinkTypeError<TransformationTypes::SinkT<double>>;
template class TransformationTypes::StorageTypeError<TransformationTypes::StorageT<double>>;
#ifdef PROVIDE_SINGLE_PRECISION
  template class TransformationTypes::CalculationError<TransformationTypes::EntryT<float,float>>;
  template class TransformationTypes::SourceTypeError<TransformationTypes::SourceT<float>>;
  template class TransformationTypes::SinkTypeError<TransformationTypes::SinkT<float>>;
  template class TransformationTypes::StorageTypeError<TransformationTypes::StorageT<float>>;
#endif


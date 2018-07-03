#include "TransformationErrors.hh"

#include "Sink.hh"
#include "Source.hh"
#include "TransformationEntry.hh"

#include <fmt/format.h>

using TransformationTypes::SinkTypeError;
using TransformationTypes::SourceTypeError;
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
SinkTypeError::SinkTypeError(const Sink *s, const std::string &message)
  : TypeError(errorMessage("sink", s, message)),
    sink(s)
{ }

/** @brief Constructor.
 *  @param s -- Source with problematic type.
 *  @param message -- error message.
 */
SourceTypeError::SourceTypeError(const Source *s, const std::string &message)
  : TypeError(errorMessage("source", s, message)),
    source(s)
{ }

/** @brief Constructor.
 *  @param e -- Entry where exception happens.
 *  @param message -- error message.
 */
CalculationError::CalculationError(const Entry *e, const std::string &message)
  : std::runtime_error(errorMessage("transformation", e, message)),
    entry(e)
{ }


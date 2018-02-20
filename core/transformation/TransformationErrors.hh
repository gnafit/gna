#pragma once

#include <string>
#include <stdexcept>

namespace TransformationTypes
{
  /**
   * @brief Base exception definition for Atypes and Rtypes classes.
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class TypeError: public std::runtime_error {
  public:
    /** @brief Constructor.
     *  @param message -- error message.
     */
    TypeError(const std::string &message)
      : std::runtime_error(message) { }
  };

  struct Sink;
  /**
   * @brief Exception to be returned from Rtypes in case of output type error.
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class SinkTypeError: public TypeError {
  public:
    SinkTypeError(const Sink *s, const std::string &message); ///< Constructor.

    const Sink *sink; ///< Sink pointer.
  };

  struct Source;
  /**
   * @brief Exception to be returned from Atypes in case of input type error.
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class SourceTypeError: public TypeError {
  public:
    SourceTypeError(const Source *s, const std::string &message); ///< Constructor.

    const Source *source; ///< Source pointer.
  };

  struct Entry;
  /**
   * @brief Exception to be returned from Rets in case of calculation error.
   * @author Dmitry Taychenachev
   * @date 2015
   */
  class CalculationError: public std::runtime_error {
  public:
    CalculationError(const Entry *e, const std::string &message); ///< Constructor.

    const Entry *entry; ///< Entry pointer.
  };
} /* TransformationTypes */

#pragma once

#include <string>
#include <stdexcept>

namespace TransformationTypes
{
  /**
   * @brief Binding exception
   * @author Maxim Gonchar
   * @date 2020
   */
  class BindError: public std::runtime_error {
  public:
    BindError(const std::string &message)
      : std::runtime_error(message) { }
  };

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

  /**
   * @brief Exception to be returned from Rtypes in case of output type error.
   * @author Dmitry Taychenachev
   * @date 2015
   */
  template<typename SinkType>
  class SinkTypeError: public TypeError {
  public:
    SinkTypeError(const SinkType *s, const std::string &message); ///< Constructor.

    const SinkType *sink; ///< Sink pointer.
  };

  /**
   * @brief Exception to be returned from Atypes in case of input type error.
   * @author Dmitry Taychenachev
   * @date 2015
   */
  template<typename SourceType>
  class SourceTypeError: public TypeError {
  public:
    SourceTypeError(const SourceType *s, const std::string &message); ///< Constructor.

    const SourceType *source; ///< Source pointer.
  };

  /**
   * @brief Exception to be returned from Itypes in case of output type error.
   * @author Maxim Gonchar
   * @date 17.07.2018
   */
  template<typename StorageType>
  class StorageTypeError: public TypeError {
  public:
    StorageTypeError(const StorageType *s, const std::string &message); ///< Constructor.

    const StorageType *storage; ///< Storage pointer.
  };

  /**
   * @brief Exception to be returned from Rets in case of calculation error.
   * @author Dmitry Taychenachev
   * @date 2015
   */
  template<typename EntryType>
  class CalculationError: public std::runtime_error {
  public:
    CalculationError(const EntryType *e, const std::string &message); ///< Constructor.

    const EntryType *entry; ///< Entry pointer.
  };
} /* TransformationTypes */

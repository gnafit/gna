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

  template<typename FloatType> struct SinkT;
  using Sink = SinkT<double>;
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

  template<typename FloatType> struct SourceT;
  using Source = SourceT<double>;
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

  template<typename FloatType> struct StorageT;
  using Storage = StorageT<double>;
  /**
   * @brief Exception to be returned from Itypes in case of output type error.
   * @author Maxim Gonchar
   * @date 17.07.2018
   */
  class StorageTypeError: public TypeError {
  public:
    StorageTypeError(const Storage *s, const std::string &message); ///< Constructor.

    const Storage *storage; ///< Storage pointer.
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

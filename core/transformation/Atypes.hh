#ifndef TYPESFUNCTIONS_H
#define TYPESFUNCTIONS_H 1

#include <string>

#include "TransformationEntry.hh"
#include "TransformationErrors.hh"
#include "Rtypes.hh"

namespace TransformationTypes
{
  /**
   * @brief Access the transformation inputs' DataType (read only).
   *
   * It's needed to:
   *   - check the consistency of the inputs in the run time.
   *   - derive the output DataType instances.
   *
   * Atypes instance is passed to each of the Entry's TypeFunction objects.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  struct Atypes {
    /**
     * @brief An exception for uninitialized Source instance
     */
    class Undefined {
    public:
      Undefined(const Source *s = nullptr) : source(s) { }
      const Source *source;
    };
    /**
     * @brief Atypes constructor.
     * @param e -- Entry instance. Atypes will get access to Entry's source types.
     */
    Atypes(const Entry *e): m_entry(e) { }

    /**
     * @brief Direct access to Sink instance, which is used as Source for the transformation.
     *
     * @param i -- Source number to return its Sink.
     * @return i-th Source's Sink instance.
     *
     * @exception Undefined in case input data is not initialized.
     */
    const Sink *sink(int i) const {
      if (!m_entry->sources[i].materialized()) {
        throw Undefined(&m_entry->sources[i]);
      }
      return m_entry->sources[i].sink;
    }

    /**
     * @brief Get i-th Source DataType (const).
     * @param i -- Source index.
     * @return i-th Source DataType.
     */
    const DataType &operator[](int i) const {
      return sink(i)->data->type;
    }

    /**
     * @brief Get number of Source instances.
     * @return number of sources.
     */
    size_t size() const { return m_entry->sources.size(); }

    static void passAll(Atypes args, Rtypes rets);     ///< Assigns shape of each input to corresponding output.

    template <size_t Arg, size_t Ret = Arg>
    static void pass(Atypes args, Rtypes rets);        ///< Assigns shape of Arg-th input to Ret-th output.

    static void ifSame(Atypes args, Rtypes rets);      ///< Checks that all inputs are of the same type (shape and content description).
    static void ifSameShape(Atypes args, Rtypes rets); ///< Checks that all inputs are of the same shape.

    template <size_t Arg>
    static void ifHist(Atypes args, Rtypes rets);      ///< Checks if Arg-th input is a histogram (DataKind=Histogram).

    template <size_t Arg>
    static void ifPoints(Atypes args, Rtypes rets);    ///< Checks if Arg-th input is an array (DataKind=Points).

    /**
     * @brief Source type exception.
     * @param dt -- incorrect DataType.
     * @param message -- exception message.
     * @return exception.
     */
    SourceTypeError error(const DataType &dt, const std::string &message = "");

    /**
     * @brief Get Entry's name
     * @return Entry's name
     */
    const std::string &name() const { return m_entry->name; }

    /**
     * @brief Empty Undefined exception.
     * @return Empty Undefined exception.
     */
    Undefined undefined() { return Undefined(); }
  private:
    const Entry *m_entry; ///< Entry instance to access Source DataType.
  };

  /**
   * @brief Assigns shape of Arg-th input to Ret-th output
   *
   * @tparam Arg -- index of Arg to read the type.
   * @tparam Ret -- index of Ret to write the type (by default Ret=Arg)
   *
   * @param args -- source types.
   * @param rets -- output types.
   *
   * @exception std::runtime_error in case of invalid index is passed.
   */
  template <size_t Arg, size_t Ret>
  inline void Atypes::pass(Atypes args, Rtypes rets) {
    if (Arg >= args.size()) {
      throw std::runtime_error("Transformation: invalid Arg index");
    }
    if (Ret >= rets.size()) {
      throw std::runtime_error("Transformation: invalid Ret index");
    }
    rets[Ret] = args[Arg];
  }

  /**
   * @brief Checks if Arg-th input is a histogram (DataKind=Histogram).
   *
   * Raises an exception otherwise.
   *
   *  @tparam Arg -- index of Arg to check.
   *
   *  @param args -- source types.
   *  @param rets -- output types.
   *
   *  @exception std::runtime_error in case input data is not a histogram.
   */
  template <size_t Arg>
  inline void Atypes::ifHist(Atypes args, Rtypes rets) {
    if (args[Arg].kind!=DataKind::Hist) {
      throw std::runtime_error("Transformation: Arg should be a histogram");
    }
  }

  /**
   * @brief Checks if Arg-th input is an array (DataKind=Points).
   *
   * Raises an exception otherwise.
   *
   * @tparam Arg -- index of Arg to check.
   *
   * @param args -- source types.
   * @param rets -- output types.
   *
   *  @exception std::runtime_error in case input data is not an array.
   */
  template <size_t Arg>
  inline void Atypes::ifPoints(Atypes args, Rtypes rets) {
    if (args[Arg].kind!=DataKind::Points) {
      throw std::runtime_error("Transformation: Arg should be an array");
    }
  }

} /* TransformationBase */

#endif

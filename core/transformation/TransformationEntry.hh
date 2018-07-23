#pragma once

#include <string>
#include <type_traits>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/noncopyable.hpp>

#include "Data.hh"
#include "Source.hh"
#include "Sink.hh"

#include "Parameters.hh"

//#include "ThreadPool.hh"


namespace TransformationTypes
{
  class Base;
  struct Args;
  struct Rets;
  struct Atypes;
  struct Rtypes;
  class InputHandle;
  class OutputHandle;

  typedef boost::ptr_vector<Source> SourcesContainer;   ///< Container for Source pointers.
  typedef boost::ptr_vector<Sink>   SinksContainer;     ///< Container for Sink pointers.

  /**
   * @brief Function, that does the actual calculation.
   *
   * This function is used to define the transformation via Entry::fun
   * and is executed via Entry::update() or Entry::touch().
   *
   * @param args -- container with transformation inputs (Args).
   * @param rets -- container with transformation outputs (Args).
   */
  typedef std::function<void(Args, Rets)> Function;

  /**
   * @brief Function, that does the input types checking and output types derivation.
   *
   * The function is used within Entry::evaluateTypes() and Entry::updateTypes().
   *
   * @param atypes -- container with transformation inputs' types (Atypes).
   * @param rtypes -- container with transformation outputs' types (Rtypes).
   */
  typedef std::function<void(Atypes, Rtypes)> TypesFunction;

  /**
   * @brief Definition of a single transformation.
   *
   * Entry defines a transformation that:
   *   - has zero or more inputs: Source instances.
   *   - has one or more outputs: Sink instances.
   *   - has a function Entry::fun that defines the transformation.
   *   - may have several type functions (Entry::typefuns), that check the input types
   *     and derive the output types.
   *
   * Entry has a taintflag (Entry::taintflag), then defines whether the Entry's Sink instances
   * contain up to date output data.
   *
   * Entry will call the transformation function Entry::fun before returning
   * Data in case Entry is tainted or any of the Inputs is tainted.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  struct Entry: public boost::noncopyable {
    Entry(const std::string &name, const Base *parent); ///< Constructor.
    Entry(const Entry &other, const Base *parent);      ///< Clone constructor.

    InputHandle addSource(const std::string &name);     ///< Initialize and return new Source.
    OutputHandle addSink(const std::string &name);      ///< Initialize and return new Sink.

    void evaluate();                                    ///< Do actual calculation by calling Entry::fun.
    void update();                                      ///< Do actual calculation by calling Entry::fun via evaluate() and resets the taintflag.
    void evaluateTypes();                               ///< Evaluate output types based on input types via Entry::typefuns call, allocate memory.
    void updateTypes();                                 ///< Evaluate output types based on input types via Entry::typefuns call, allocate memory.

    void touch();                                       ///< Update the transformation if it is not frozen and tainted.
    const Data<double> &data(int i);                    ///< Evaluates the function if needed and returns i-th data.

    void freeze() { frozen = true; }                    ///< Freeze the Entry. While entry is frozen the taintflag is not propagated. Entry is always up to date.
    void unfreeze() { frozen = false; }                 ///< Unfreeze the Entry. Enables the taintflag propagation.

    void mark_running() {running = true; }
    void mark_not_running() {running = false; }
 

    bool check() const;                                 ///< Checks that Data are initialized.
    void dump(size_t level = 0) const;                  ///< Recursively print Source names and their connection status.

    std::string name;                                   ///< Transformation name.
    std::string label;                                  ///< Transformation label.
    SourcesContainer sources;                           ///< Transformation inputs (sources).
    SinksContainer sinks;                               ///< Transformation outputs (sinks).
    Function fun=nullptr;                               ///< The function that does actual calculation.
    std::vector<TypesFunction> typefuns;                ///< Vector of TypeFunction objects.
    taintflag tainted;                                  ///< taintflag shows whether the result is up to date.
    const Base *parent;                                 ///< Base class, containing the transformation Entry.
    int initializing;                                   ///< Initialization status. initializing>0 when Entry is being configured via Initializer.
    bool frozen;                                        ///< If Entry is frozen, it is not updated even if tainted.
    bool running=false;                                 ///< If Entry is running, it is locked in multithreading case.

  private:
    template <typename InsT, typename OutsT>
    void initSourcesSinks(const InsT &inputs, const OutsT &outputs); ///< Initialize the clones for inputs and outputs.
  }; /* struct Entry */

} /* TransformationTypes */

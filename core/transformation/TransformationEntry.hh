#pragma once

#include <string>
#include <type_traits>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/noncopyable.hpp>

#include "Data.hh"
#include "Parameters.hh"
#include "TransformationFunction.hh"
#include "FunctionDescriptor.hh"

#include "Source.hh"
#include "Sink.hh"
#include "Storage.hh"

#ifdef GNA_CUDA_SUPPORT
#include "DataLocation.hh"
#include "cuda_config_vars.h"
#endif


namespace TransformationTypes
{
  class Base;
  class InputHandle;
  class OutputHandle;

  typedef boost::ptr_vector<Source>  SourcesContainer;   ///< Container for Source pointers.
  typedef boost::ptr_vector<Sink>    SinksContainer;     ///< Container for Sink pointers.
  typedef boost::ptr_vector<Storage> StoragesContainer;  ///< Container for Storage pointers.

  /**
   * @brief Definition of a single transformation.
   *
   * Entry defines a transformation that:
   *   - has zero or more inputs: Source instances.
   *   - has one or more outputs: Sink instances.
   *   - has a function Entry::fun that defines the transformation.
   *   - may have several type functions (Entry::typefuns), that check the
   *     input types, derive the output types and the types of internal data to be allocated.
   *
   * Entry has a taintflag (Entry::taintflag), then defines whether the Entry's Sink instances
   * contain up to date output data.
   *
   * Entry will call the transformation function Entry::fun before returning
   * Data in case Entry is tainted or any of the Inputs is tainted.
   *
   * Entry::fun may have several different implementations that can be used for example to define
   * the GPU version of the transformation. The implementations are named and
   * stored in the Entry::functions container. The active function Entry::fun may be switched to any
   * function in Entry::functions via Entry::switchFunction() method.
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  struct Entry: public boost::noncopyable {
    Entry(const std::string &name, const Base *parent);  ///< Constructor.
    Entry(const Entry &other, const Base *parent);       ///< Clone constructor.

    InputHandle addSource(const std::string &name);      ///< Initialize and return new Source.
    OutputHandle addSink(const std::string &name);       ///< Initialize and return new Sink.

    void evaluate();                                    ///< Do actual calculation by calling Entry::fun.
    void update();                                      ///< Do actual calculation by calling Entry::fun via evaluate() and resets the taintflag.
    void evaluateTypes();                               ///< Evaluate output types based on input types via Entry::typefuns call, allocate memory.
    void updateTypes();                                 ///< Evaluate output types based on input types via Entry::typefuns call, allocate memory.

    void touch();                                        ///< Update the transformation if it is not frozen and tainted.
    const Data<double> &data(int i);                     ///< Evaluates the function if needed and returns i-th data.

    void freeze() { frozen = true; }                     ///< Freeze the Entry. While entry is frozen the taintflag is not propagated. Entry is always up to date.
    void unfreeze() { frozen = false; }                  ///< Unfreeze the Entry. Enables the taintflag propagation.

    bool check() const;                                  ///< Checks that Data are initialized.
    void dump(size_t level = 0) const;                   ///< Recursively print Source names and their connection status.

#ifdef GNA_CUDA_SUPPORT
    void setEntryLocation(DataLocation loc) { 		///< Sets the target (Host or Device) for execution of current transformation
		m_entryLoc = loc; 
    }
    DataLocation getEntryLocation() const { 		///<  Returns the target (Host or Device) for execution of current transformation
		return m_entryLoc; 
    }
#endif

    std::string name;                                   ///< Transformation name.
    std::string label;                                  ///< Transformation label.
    const Base *parent;                                 ///< Base class, containing the transformation Entry.


    // Data
    SourcesContainer sources;                            ///< Transformation inputs (sources).
    SinksContainer sinks;                                ///< Transformation outputs (sinks).
    StoragesContainer storages;                          ///< Transformation internal Storage instances.

    // Functions
    Function fun=nullptr;                                ///< The function that does actual calculation.
    TypesFunctionsContainer typefuns;                    ///< Vector of TypeFunction objects.
    FunctionDescriptorsContainer functions;              ///< Map with FunctionDescriptor instances, containing several Function implementations.
    std::string funcname;                                ///< Active Function name.

    // Status
    taintflag tainted;                                   ///< taintflag shows whether the result is up to date.
    int initializing;                                    ///< Initialization status. initializing>0 when Entry is being configured via Initializer.
    bool frozen;                                         ///< If Entry is frozen, it is not updated even if tainted.

    void switchFunction(const std::string& name);        ///< Use Function `name` as Entry::fun.
  private:
#ifdef GNA_CUDA_SUPPORT
    DataLocation m_entryLoc = DataLocation::Host;	///< In case of GPU support is swiched on, the target for execution(Host or Device). Host by default.
#endif
    template <typename InsT, typename OutsT>
    void initSourcesSinks(const InsT &inputs, const OutsT &outputs); ///< Initialize the Data for inputs and outputs.
    void initInternals(StorageTypesFunctionArgs& fargs);             ///< Initialize the Data for the internal storage.

  }; /* struct Entry */

} /* TransformationTypes */

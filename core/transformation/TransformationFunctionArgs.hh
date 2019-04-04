#pragma once

#include <memory>
#include "Args.hh"
#include "Rets.hh"
#include "Ints.hh"
#include "Atypes.hh"
#include "Rtypes.hh"
#include "Itypes.hh"

namespace TransformationTypes
{
  template<typename FloatType,typename SizeType> struct GPUFunctionArgsT;
  template<typename SourceType,typename SinkType> class EntryT;

  /**
   * @brief Transformation Function arguments.
   *
   * FunctionArgs instance is pased to the Function and contains the necessary data:
   *   - inputs;
   *   - outputs;
   *   - internal storage.
   *
   * Inputs are available read-only.
   *
   * @author Maxim Gonchar
   * @date 07.2018
   */
  template<typename SourceFloatType, typename SinkFloatType>
  class FunctionArgsT {
  public:
    using EntryType              = EntryT<SourceFloatType,SinkFloatType>;
    using FunctionArgsType       = FunctionArgsT<SourceFloatType,SinkFloatType>;
    using GPUFunctionArgsType    = GPUFunctionArgsT<SourceFloatType, size_t>;
    using GPUFunctionArgsPtr     = std::unique_ptr<GPUFunctionArgsType>;
    using ArgsType               = ArgsT<SourceFloatType,SinkFloatType>;
    using RetsType               = RetsT<SourceFloatType,SinkFloatType>;
    using IntsType               = IntsT<SourceFloatType,SinkFloatType>;

    FunctionArgsT(EntryType* e) : args(e), rets(e), ints(e), m_entry(e) {  }         ///< Constructor.
    FunctionArgsT(const FunctionArgsType& other) : FunctionArgsT(other.m_entry) {  } ///< Copy constructor.
    ~FunctionArgsT();

    ArgsType args; ///< arguments, or transformation inputs (read-only)
    RetsType rets; ///< return values, or transformation outputs (writable)
    IntsType ints; ///< preallocated data arrays for the transformation's internal usage (writable)

    GPUFunctionArgsPtr gpu; ///< GPU function arguments

    void requireGPU();                    ///< Initialize GPU function arguments
    void updateTypes();                   ///< Update arguments and types

    size_t getMapping(size_t input){ return m_entry->mapping[input]; }
  private:
    EntryType *m_entry; ///< Entry instance to access Sinks.
  };

  /**
   * @brief Transformation TypesFunction arguments.
   *
   * TypesFunctionArgs instance is pased to the typesFunction and contains the necessary data types:
   *   - inputs' data types;
   *   - outputs' data types;
   *   - internal storages' data types.
   *
   * The input data types are available read-only.
   *
   * @author Maxim Gonchar
   * @date 07.2018
   */
  template<typename SourceFloatType, typename SinkFloatType>
  struct TypesFunctionArgsT {
    using EntryType = EntryT<SourceFloatType,SinkFloatType>;
    using Atypes = AtypesT<SourceFloatType,SinkFloatType>;
    using Rtypes = RtypesT<SourceFloatType,SinkFloatType>;
    using Itypes = ItypesT<SourceFloatType,SinkFloatType>;
    TypesFunctionArgsT(EntryType* e) : args(e), rets(e), ints(e) {  } ///< Constructor.

    Atypes args; ///< arguments'/inputs' data types (read-only)
    Rtypes rets; ///< return values'/outputs' data  types (writable)
    Itypes ints; ///< preallocated storage's data types (writable)
  };

  /**
   * @brief Transformation StorageTypesFunction arguments.
   *
   * TypesFunctionArgs instance is pased to the typesFunction and contains the necessary data types:
   *   - inputs' data types;
   *   - outputs' data types;
   *   - internal storages' data types.
   *
   * The input and output data types are available read-only. The StorageTypesFunction may only create
   * and modify the information about the internal storage requirements.
   *
   * @author Maxim Gonchar
   * @date 07.2018
   */
  template<typename SourceFloatType, typename SinkFloatType>
  struct StorageTypesFunctionArgsT {
    using TypesFunctionArgsType = TypesFunctionArgsT<SourceFloatType,SinkFloatType>;
    StorageTypesFunctionArgsT(TypesFunctionArgsType& fargs) : args(fargs.args), rets(fargs.rets), ints(fargs.ints) {  } ///< Constructor.

    AtypesT<SourceFloatType,SinkFloatType>& args;       ///< arguments'/inputs' data types (read-only)
    const RtypesT<SourceFloatType,SinkFloatType>& rets; ///< return values'/outputs' data  types (read-only)
    ItypesT<SourceFloatType,SinkFloatType>& ints;       ///< preallocated storage's data types (writable)
  };
}


#pragma once

#include <boost/noncopyable.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/optional.hpp>

#include "Accessor.hh"
#include "TransformationEntry.hh"
#include "Exceptions.hh"

/**
 * @brief A namespace for transformations.
 * The namespace defines Entry, Sink, Source and Base classes, necessary to deal
 * with transformations. Helper classes are also provided here.
 * @author Dmitry Taychenachev
 * @date 2015
 */
namespace TransformationTypes {
  template<typename T,typename SourceFloatType, typename SinkFloatType>
  class InitializerT;

  /**
   * @brief Base transformation class handling.
   *
   * Base class is derived by GNAObject does the bookkeeping for the transformations and defines the GNAObject transformation handling.
   *
   * Base class defines an object containing several transformation Entry instances.
   *
   * Each Entry defines an elementary transformation that will be updated in case any of Entry's inputs is updated.
   * Base enables the user to organize a more complex transformation each part of which depends on its own inputs
   * and thus may be updated independently. Entry instances within Base class may share internal data directly
   * (not via Sink-Source connections).
   *
   * @author Dmitry Taychenachev
   * @date 2015
   */
  template<typename SourceFloatType, typename SinkFloatType>
  class BaseT: public boost::noncopyable {
    template <typename T,typename SourceFloatType1, typename SinkFloatType1>
    friend class InitializerT;
    friend class TransformationDescriptor;
    template <typename SourceFloatType1, typename SinkFloatType1>
    friend class AccessorT;
  public:
    using BaseType       = BaseT<SourceFloatType,SinkFloatType>;
    using AccessorType   = AccessorT<SourceFloatType,SinkFloatType>;
    using EntryType      = EntryT<SourceFloatType,SinkFloatType>;
    using EntryContainerType = boost::ptr_vector<EntryType>;

    BaseT(const BaseType &other);                                         ///< Clone constructor.
    //BaseT &operator=(const BaseType &other);                              ///< Clone assignment.
    virtual ~BaseT(){}

  protected:
    BaseT(): t_(*this) { }                                                ///< Default constructor.
    /**
     * @brief Constructor that limits the maximal number of allowed Entry instances.
     * @param maxentries -- maximal number of Entry instances the Base may keep.
     */
    BaseT(size_t maxentries): BaseT() {
      m_maxEntries = maxentries;
    }

    /**
     * @brief Get Entry by index.
     * @param idx -- index of an Entry to return.
     * @return Entry.
     */
    EntryType &getEntry(size_t idx) {
      return m_entries[idx];
    }
    EntryType &getEntry(const std::string &name);                        ///< Get an Entry by name.

    AccessorType t_;                                                     ///< An Accessor to Base's Entry instances via Handle.

    size_t addEntry(EntryType *e);                                       ///< Add new Entry.
    EntryContainerType m_entries;                                        ///< Vector of Entry pointers. Calls destructors when deleted.
    boost::optional<size_t> m_maxEntries;                                ///< Maximum number of allowed entries.
    //void copyEntries(const BaseType &other);                             ///< Clone entries from the other Base.
  }; /* class Base */
} /* namespace TransformationTypes */


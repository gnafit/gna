#pragma once

#include <boost/noncopyable.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/optional.hpp>

#include "Accessor.hh"
#include "TransformationEntry.hh"
#include "Exceptions.hh"

class GNAObject;
template <typename Derived>
class TransformationBind;

/**
 * @brief A namespace for transformations.
 * The namespace defines Entry, Sink, Source and Base classes, necessary to deal
 * with transformations. Helper classes are also provided here.
 * @author Dmitry Taychenachev
 * @date 2015
 */
namespace TransformationTypes {
  template<typename T>
  class Initializer;

  typedef boost::ptr_vector<Entry> EntryContainer; ///< Container for Entry pointers.

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
  class Base: public boost::noncopyable {
    template <typename T>
    friend class ::TransformationBind;
    template <typename T>
    friend class Initializer;
    friend class TransformationDescriptor;
    friend class Accessor;
    friend class ::GNAObject;
  public:
    Base(const Base &other);                                             ///< Clone constructor.
    Base &operator=(const Base &other);                                  ///< Clone assignment.
  protected:
    Base(): t_(*this) { }                                                ///< Default constructor.
    /**
     * @brief Constructor that limits the maximal number of allowed Entry instances.
     * @param maxentries -- maximal number of Entry instances the Base may keep.
     */
    Base(size_t maxentries): Base() {
      m_maxEntries = maxentries;
    }

    // Not implemented!
    // void connect(Source &source, Base *sinkobj, Sink &sink);

    /**
     * @brief Get Entry by index.
     * @param idx -- index of an Entry to return.
     * @return Entry.
     */
    Entry &getEntry(size_t idx) {
      return m_entries[idx];
    }
    Entry &getEntry(const std::string &name);                            ///< Get an Entry by name.

    Accessor t_;                                                         ///< An Accessor to Base's Entry instances via Handle.
  private:
    size_t addEntry(Entry *e);                                           ///< Add new Entry.
    boost::ptr_vector<Entry> m_entries;                                  ///< Vector of Entry pointers. Calls destructors when deleted.
    boost::optional<size_t> m_maxEntries;                                ///< Maximum number of allowed entries.
    void copyEntries(const Base &other);                                 ///< Clone entries from the other Base.
  }; /* class Base */

} /* namespace TransformationTypes */


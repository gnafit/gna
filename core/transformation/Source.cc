#include "Source.hh"
#include "TransformationEntry.hh"

#include <stdexcept>

using TransformationTypes::SinkT;
using TransformationTypes::SourceT;

/**
 * @brief Connect the Source to the Sink.
 *
 * After the connection:
 *   - the current Entry is subscribed to the Sink's taintflag to track dependencies.
 *   - Execute Sink::touchTypes() is called to define the input data types if not already defined.
 *   - Execute Entry::updateTypes() to check input data types and derive output data types if needed.
 *
 * @param newsink -- sink to connect to.
 *
 * @exception std::runtime_error if there already exists a connected Sink.
 */
template<typename FloatType>
void SourceT<FloatType>::connect(SinkT<FloatType> *newsink) {
  if (sink) {
    std::cerr << this << " " << name << " " << sink->entry->name << "\n";
    throw std::runtime_error(
      (fmt::format("Transformation: source `{0}' is already connected to sink `{1}',"
              " won't connect to `{2}'", name, sink->name, newsink->name)));
  }
  //if (false) {
    //throw std::runtime_error("Transformation: connecting incompatible types");
  //}
  TR_DPRINTF("connecting source `%s'[%p] on `%s' to sink `%s'[%p] on `%s'\n", name.c_str(), (void*)this, entry->name.c_str(), newsink->name.c_str(), (void*)newsink, newsink->entry->name.c_str());
  sink = newsink;
  if( !this->inactive ){
    sink->entry->tainted.subscribe(entry->tainted);
  }
  newsink->sources.push_back(this);
  try {
    newsink->entry->touchTypes();
    entry->updateTypes();
  } catch (const std::exception &exc) {
    std::cerr << "exception in types calculation: ";
    std::cerr << exc.what() << "\n";
    std::terminate();
  }
}

template struct TransformationTypes::SourceT<double>;
#ifdef PROVIDE_SINGLE_PRECISION
  template struct TransformationTypes::SourceT<float>;
#endif

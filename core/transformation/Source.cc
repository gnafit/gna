#include "Source.hh"
#include "TransformationEntry.hh"

#include <stdexcept>

using TransformationTypes::Source;
using TransformationTypes::Sink;

/**
 * @brief Connect the Source to the Sink.
 *
 * After the connection:
 *   - the current Entry is subscribed to the Sink's taintflag to track dependencies.
 *   - Execute Sink::evaluateTypes() is called to define the input data types if not already defined.
 *   - Execute Entry::evaluateTypes() to check input data types and derive output data types if needed.
 *
 * @param newsink -- sink to connect to.
 *
 * @exception std::runtime_error if there already exists a connected Sink.
 */
void Source::connect(Sink *newsink) {
  if (sink) {
    std::cerr << this << " " << name << " " << sink->entry->name << "\n";
    throw std::runtime_error(
      (format("Transformation: source `%1%' is already connected to sink `%2%',"
              " won't connect to `%3%'") % name % sink->name % newsink->name)
       .str()
      );
  }
  //if (false) {
    //throw std::runtime_error("Transformation: connecting incompatible types");
  //}
  TR_DPRINTF("connecting source `%s'[%p] on `%s' to sink `%s'[%p] on `%s'\n", name.c_str(), (void*)this, entry->name.c_str(), newsink->name.c_str(), (void*)newsink, newsink->entry->name.c_str());
  sink = newsink;
  sink->entry->tainted.subscribe(entry->tainted);
  newsink->sources.push_back(this);
  try {
    newsink->entry->evaluateTypes();
    entry->evaluateTypes();
  } catch (const std::exception &exc) {
    std::cerr << "exception in types calculation: ";
    std::cerr << exc.what() << "\n";
    std::terminate();
  }
}


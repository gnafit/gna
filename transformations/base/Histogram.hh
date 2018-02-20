#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>
#include "GNAObject.hh"

/**
 * @brief Transformation object holding a static 1-dimensional histogram.
 *
 * Outputs:
 *   - `hist.hist` - 1-dimensional histogram with fixed data.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
class Histogram: public GNASingleObject,
                 public TransformationBlock<Histogram> {
public:
  /**
   * @brief Construct 1d histogram from two arrays: edges and data
   * @param nbins - number of bins in a histogram.
   * @param edegs - pointer to an array with nbins+1 edges.
   * @param edegs - pointer to an array with nbins data points.
   */
  Histogram(size_t nbins, const double *edges, const double *data)
    : m_edges(edges, edges+nbins+1), m_data(Eigen::Map<const Eigen::ArrayXd>(data, nbins))
  {
    init();
  }

  /**
   * @brief Return the vector with histogram edges.
   * @return vector with histogram edges.
   */
  const std::vector<double> &edges() const { return m_edges; }

  /**
   * @brief Return std::vector with a copy of data.
   *
   * @deprecated This method is kept for backwards compatibility.
   *
   * @return std::vector with copy of data.
   */
  std::vector<double> data() const { return std::vector<double>(m_data.data(), m_data.data()+m_data.size()); }

  /**
   * @brief Return the size of an array.
   * @return number of elements in an array.
   */
  size_t size() const {
    return m_data.size();
  }

  /**
   * @brief Return the pointer to C++ array.
   * @return array pointer.
   */
  const double *ptr() const {
    return m_data.data();
  }

protected:
  void init() {
    transformation_(this, "hist")                               /// Initialize the transformation hist.
      .output("hist")                                           /// Add an output hist.
      .types([](Histogram *obj, Atypes /*args*/, Rtypes rets) { /// Define the TypesFunction:
          rets[0] = DataType().hist().edges(obj->edges());      ///   - assign the data shape and bin edges for the first output (hist).
          rets[0].preallocated(obj->m_data.data());             ///   - tell the DataType that the buffer is preallocated (m_data).
        })
      .func([](Args,Rets) {})                                   /// Assign empty Function.
      .finalize();                                              /// Tell the initializer that there are no more configuration and it may initialize the types.
  }
  std::vector<double> m_edges;                                  ///< Vector with bin edges.
  Eigen::ArrayXd m_data;                                        ///< The array holding the raw 1d data buffer.
};

#endif // HISTOGRAM_H

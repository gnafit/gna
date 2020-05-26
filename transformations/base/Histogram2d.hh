#pragma once

#include <vector>
#include "GNAObject.hh"

/**
 * @brief Transformation object holding a static 2-dimensional histogram.
 *
 * Outputs:
 *   - `hist.hist` - 2-dimensional histogram with fixed data.
 *
 * @author Dmitry Taychenachev
 * @date 2015
 */
class Histogram2d: public GNASingleObject,
                   public TransformationBind<Histogram2d> {
public:
  /**
   * @brief Construct 2d histogram from two arrays: edges and data
   * @param xbins - number of X bins in a histogram.
   * @param xedges - pointer to an array with nbins+1 edges.
   * @param ybins - number of Y bins in a histogram.
   * @param yedges - pointer to an array with nbins+1 edges.
   * @param data - pointer to an array with data points of shape Xbins x Ybins (column-major).
   */
  Histogram2d(size_t xbins, const double *xedges, size_t ybins, const double *yedges, const double *data);

  /**
   * @brief Return the vector with histogram X edges.
   * @return vector with histogram edges.
   */
  const std::vector<double> &xedges() const { return m_xedges; }

  /**
   * @brief Return the vector with histogram X edges.
   * @return vector with histogram edges.
   */
  const std::vector<double> &yedges() const { return m_yedges; }

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
    transformation_("hist")                                      /// Initialize the transformation hist.
      .output("hist")                                            /// Add an output hist.
      .types([](Histogram2d *obj, TypesFunctionArgs& fargs) {    /// Define the TypesFunction:
          fargs.rets[0] = DataType().hist().edges(obj->xedges(), obj->yedges()); ///   - assign the data shape and bin edges for the first output (hist).
          /* fargs.rets[0].preallocated(obj->m_data.data());        ///   - tell the DataType that the buffer is preallocated (m_data). */
        })
      .func([](Histogram2d* obj, FunctionArgs& fargs) {
              fargs.rets[0].arr2d = obj->m_data;})
      .finalize();                                               /// Tell the initializer that there are no more configuration and it may initialize the types.
  }
  std::vector<double> m_xedges;                                  ///< Vector with X bin edges.
  std::vector<double> m_yedges;                                  ///< Vector with Y bin edges.
  Eigen::ArrayXXd m_data;                                 ///< Array with raw bin content.
};

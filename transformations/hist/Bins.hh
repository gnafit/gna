#pragma once

#include <vector>
#include "GNAObject.hh"

/**
 * @brief Transformation object holding a static 1-dimensional array with bin edges, array with bin centers and array with bins widths.
 *
 * Outputs:
 *   - `bins.edges` - bin edges
 *   - `bins.centers` - bin centers
 *   - `bins.widths` - bin widths
 *
 * @author Maxim Gonchar
 * @date 2017
 */
class Bins: public GNASingleObject,
            public TransformationBind<Bins> {
public:
  /**
   * @brief Construct 1d array from a vector of doubles.
   *
   * The size is determined by the size of the vector.
   *
   * @param edges - vector of doubles.
   */
  Bins(const std::vector<double> &edges) : Bins(&edges[0], edges.size()) {  }

  /**
   * @brief Construct 1d array from C++ array of doubles.
   * @param edges - pointer to an array of doubles.
   * @param bins - number of bins.
   */
  Bins(const double *edges, size_t bins)
    : m_edges(Eigen::Map<const Eigen::ArrayXd>(edges, bins+1))
  {
    init();
  }

protected:
  /**
   * @brief Initialize the transformation.
   *
   * Transformation `bins` with three outputs `edges`, `centers` and `widths`.
   */
  void init() {
    /// Check that at least two bin edges were passed
    if(m_edges.size()<2){
      throw std::runtime_error("Bins transformation should be initialized with at least two edges.");
    }
    transformation_("bins")                                               /// Initialize the transformation points.
      .output("edges")                                                    /// Add an output for edges.
      .output("centers")                                                  /// Add an output for centers.
      .output("widths")                                                   /// Add an output for widths.
      .types([](Bins *obj, TypesFunctionArgs& fargs) {                    /// Define the TypesFunction:
          auto nedges=obj->m_edges.size();                                ///   - get the number of edges.
          auto nbins=nedges-1;                                            ///   - make the number of bins.
          fargs.rets[0] = DataType().points().shape(nedges);              ///   - assign the data shape for the first output (edges).
          fargs.rets[0].preallocated(obj->m_edges.data());                ///   - tell the DataType that the buffer is preallocated (m_edges).
          fargs.rets[1] = DataType().points().shape(nbins);               ///   - assign the data shape for the second output (centers).
          fargs.rets[2] = DataType().points().shape(nbins);               ///   - assign the data shape for the third output (widths).
        })
    .func([](FunctionArgs& fargs){                                        /// Assign Function:
          auto& rets = fargs.rets;                                        ///   - store return values
          auto& edges   = rets[0].x;                                      ///   - store edges
          auto& centers = rets[1].x;                                      ///   - store centers
          auto& widths  = rets[2].x;                                      ///   - store widths
          auto nbins = edges.size()-1;                                    ///   - store n bins
          centers = edges.head(nbins)+edges.tail(nbins);                  ///   - calculate centers
          centers*=0.5;                                                   ///   - (centers)
          widths = edges.tail(nbins)-edges.head(nbins);                   ///   - calcualte widths
          })
    .finalize();                                                        /// Tell the initializer that there are no more configuration and it may initialize the types.
  }
  Eigen::ArrayXd m_edges;                                                ///< The array holding the raw 1d data buffer.
};

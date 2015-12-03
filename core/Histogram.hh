#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>

#include "GNAObject.hh"

class Histogram: public GNAObject,
                 public Transformation<Histogram> {
public:
  Histogram(size_t nbins, const double *edges, const double *data)
    : m_edges(edges, edges+nbins+1), m_data(data, data+nbins)
  { }
  const std::vector<double> &edges() const { return m_edges; }
  const std::vector<double> &data() const { return m_data; }
protected:
  void init() {
    transformation_(this, "hist")
      .output("hist")
      .types([](Histogram *obj, Atypes /*args*/, Rtypes rets) {
          rets[0] = DataType().hist().edges(obj->edges());
        })
      .func([](Histogram *obj, Args /*args*/, Rets rets) {
          auto &pts = obj->data();
          rets[0].x = Eigen::Map<const Eigen::ArrayXd>(&pts[0], pts.size());
        });
  }
  std::vector<double> m_edges;
  std::vector<double> m_data;
};

#endif // HISTOGRAM_H

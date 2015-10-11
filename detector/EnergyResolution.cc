#include <boost/math/constants/constants.hpp>

#include "EnergyResolution.hh"

const double pi = boost::math::constants::pi<double>();

EnergyResolution::EnergyResolution() {
  variable_(&m_a, "Eres_a");
  variable_(&m_b, "Eres_b");
  variable_(&m_c, "Eres_c");
  callback_([this] { fillCache(); });

  using namespace std::placeholders;
  transformation_(this, "smear")
    .input("Nvis", DataType().hist().any())
    .output("Nrec", DataType().hist().any())
    .types(Atypes::pass<0>,
           [](EnergyResolution *obj, Atypes args, Rtypes /*rets*/) {
             obj->m_datatype = args[0];
             obj->fillCache();
           })
    .func(&EnergyResolution::calcSmear);
}

double EnergyResolution::relativeSigma(double Etrue) {
  return sqrt(pow(m_a, 2)+ pow(m_b, 2)/Etrue + pow(m_c, 2)/pow(Etrue, 2));
}

double EnergyResolution::resolution(double Etrue, double Erec) {
  static const double twopisqr = std::sqrt(2*pi);
  double sigma = Etrue * relativeSigma(Etrue);
  double reldiff = (Etrue - Erec)/sigma;

  return std::exp(-0.5*pow(reldiff, 2))/(twopisqr*sigma);
}

void EnergyResolution::fillCache() {
  m_size = m_datatype.hist().bins();
  if (m_size == 0) {
    return;
  }
  m_rescache.resize(m_size*m_size);
  m_cacheidx.resize(m_size + 1);
  m_startidx.resize(m_size);
  std::vector<double> buf(m_size);

  int cachekey = 0;
  for (size_t j = 0; j < m_size; ++j) {
    double Etrue = (m_datatype.edges[j+1] + m_datatype.edges[j])/2;
    double dEtrue = m_datatype.edges[j+1] - m_datatype.edges[j];

    int startidx = -1;
    int endidx = -1;
    for (size_t i = 0; i < m_size; ++i) {
      double Erec = (m_datatype.edges[i+1] + m_datatype.edges[i])/2;
      double rEvents = dEtrue*resolution(Etrue, Erec);

      if (rEvents < 1E-10) {
        if (startidx >= 0) {
          endidx = i;
          break;
        }
        continue;
      }
      buf[i] = rEvents;
      if (startidx < 0) {
        startidx = i;
      }
    }
    if (endidx < 0) {
      endidx = m_size;
    }
    if (startidx >= 0) {
      std::copy(&buf[startidx], &buf[endidx], &m_rescache[cachekey]);
      m_cacheidx[j] = cachekey;
      cachekey += endidx - startidx;
    } else {
      m_cacheidx[j] = cachekey;
    }
    m_startidx[j] = startidx;
  }
  m_cacheidx[m_size] = cachekey;
}

void EnergyResolution::calcSmear(Args args, Rets rets) {
  const double *events_true = args[0].x.data();
  double *events_rec = rets[0].x.data();

  size_t insize = args[0].type.size();
  size_t outsize = rets[0].type.size();

  std::copy(events_true, events_true + insize, events_rec);
  double loss = 0.0;
  for (size_t j = 0; j < insize; ++j) {
    double *cache = &m_rescache[m_cacheidx[j]];
    int startidx = m_startidx[j];
    int cnt = m_cacheidx[j+1] - m_cacheidx[j];
    for(int off = 0; off < cnt; ++off) {
      size_t i = startidx + off;
      if (i == j) {
        continue;
      }
      double rEvents = cache[off];
      double delta = rEvents*events_true[j];

      events_rec[i] += delta;
      int inv = 2*j - i;
      if (inv < 0 || (size_t)inv >= outsize) {
        events_rec[j] -= delta;
        loss += delta;
      }
      events_rec[j] -= delta;
    }
  }
}

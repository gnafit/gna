#include <boost/math/constants/constants.hpp>
#include "EnergyResolution.hh"
#include "iostream"

constexpr double pi = boost::math::constants::pi<double>();

/* TODO: reimplement it with some good matrix machinery, maybe Eigen? */
EnergyResolution::EnergyResolution() {
  variable_(&m_a, "Eres_a");
  variable_(&m_b, "Eres_b");
  variable_(&m_c, "Eres_c");
  callback_([this] { fillCache(); });

  using namespace std::placeholders;
  transformation_(this, "smear")
    .input("Nvis")
    .output("Nrec")
    .types(Atypes::pass<0>,
           [](EnergyResolution *obj, Atypes args, Rtypes /*rets*/) {
             obj->m_datatype = args[0];
             obj->fillCache();
           })
    .func(&EnergyResolution::calcSmear);
}

double EnergyResolution::relativeSigma(double Etrue) {
  return sqrt(pow(m_a, 2)+ pow(m_b, 2)/Etrue + pow(m_c/Etrue, 2));
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
  /* fill the cache matrix with probalilities for number of events to leak to other bins */
  /* colums corressponds to reconstrucred energy and rows to true energy */
  for (size_t etrue_row = 0; etrue_row < m_size; ++etrue_row) {
    double Etrue = (m_datatype.edges[etrue_row+1] + m_datatype.edges[etrue_row])/2;
    double dEtrue = m_datatype.edges[etrue_row+1] - m_datatype.edges[etrue_row];

    int startidx = -1;
    int endidx = -1;
    /* precalculating probabilities for events in given bin to leak to 
     * neighbor bins  */
    for (size_t erec_column = 0; erec_column < m_size; ++erec_column) {
      double Erec = (m_datatype.edges[erec_column+1] + m_datatype.edges[erec_column])/2;
      double rEvents = dEtrue*resolution(Etrue, Erec);

      if (rEvents < 1E-10) {
        if (startidx >= 0) {
           endidx = erec_column;
           break;
        }
        continue;
      }
      buf[erec_column] = rEvents;
      if (startidx < 0) {
        startidx = erec_column;
      }
    }
    if (endidx < 0) {
      endidx = m_size;
    }
    /* after row is filled it's time to move it to cache */
    if (startidx >= 0) {
      /* std::copy(&buf[startidx], &buf[endidx], &m_rescache[cachekey]);  */
      std::copy(std::make_move_iterator(&buf[startidx]),
                std::make_move_iterator(&buf[endidx]), 
                &m_rescache[cachekey]); 

      /* put the cachekey into index storage and update it to new value */
      m_cacheidx[etrue_row] = cachekey;
      cachekey += endidx - startidx;
    } else {
       /* if no values cached cachekey remains the same  */
     m_cacheidx[etrue_row] = cachekey;
    }
    m_startidx[etrue_row] = startidx;
  }
  /* nothing is stored at the end, huh? */
  m_cacheidx[m_size] = cachekey;
}

/* Apply precalculated cache and actually smear */
void EnergyResolution::calcSmear(Args args, Rets rets) {
  const double *events_true = args[0].x.data();
  double *events_rec = rets[0].x.data();

  size_t insize = args[0].type.size();
  size_t outsize = rets[0].type.size();
  /* assert(insize == outsize);  */

  std::copy(events_true, events_true + insize, events_rec);
  double loss = 0.0;
  for (size_t etrue_row = 0; etrue_row < insize; ++etrue_row) {
     /* get the cache line */
    double *cache = &m_rescache[m_cacheidx[etrue_row]];
    int startidx = m_startidx[etrue_row];
     /* get number of valid cache entries for corresponding Etrue  */
    int cnt = m_cacheidx[etrue_row+1] - m_cacheidx[etrue_row];
    for(int off = 0; off < cnt; ++off) {
      /* current position in cache */
      size_t erec_column = startidx + off;
      if (erec_column == etrue_row) {
        continue;
      }
      double rEvents = cache[off];
      double delta = rEvents*events_true[etrue_row];

      events_rec[erec_column] += delta;
      int inv = 2*etrue_row - erec_column;
      /* if get outside the bins of histo then events are LOST */
      if (inv < 0 || (size_t)inv >= outsize) {
        events_rec[etrue_row] -= delta;
        loss += delta;
      }
      events_rec[etrue_row] -= delta;
    }
  }
}

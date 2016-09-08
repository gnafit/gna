#include <boost/math/constants/constants.hpp>
#include "EnergyResolution.hh"

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

double EnergyResolution::relativeSigma(double Etrue) const noexcept{
  return sqrt(pow(m_a, 2)+ pow(m_b, 2)/Etrue + pow(m_c/Etrue, 2));
}

double EnergyResolution::resolution(double Etrue, double Erec) const noexcept {
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
  for (size_t etrue = 0; etrue < m_size; ++etrue) {
    double Etrue = (m_datatype.edges[etrue+1] + m_datatype.edges[etrue])/2;
    double dEtrue = m_datatype.edges[etrue+1] - m_datatype.edges[etrue];

    int startidx = -1;
    int endidx = -1;
    /* precalculating probabilities for events in given bin to leak to 
     * neighbor bins  */
    for (size_t erec = 0; erec < m_size; ++erec) {
      double Erec = (m_datatype.edges[erec+1] + m_datatype.edges[erec])/2;
      double rEvents = dEtrue*resolution(Etrue, Erec);

      if (rEvents < 1E-10) {
        if (startidx >= 0) {
           endidx = erec;
           break;
        }
        continue;
      }
      buf[erec] = rEvents;
      if (startidx < 0) {
        startidx = erec;
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
      m_cacheidx[etrue] = cachekey;
      cachekey += endidx - startidx;
    } else {
       /* if no values cached cachekey remains the same  */
     m_cacheidx[etrue] = cachekey;
    }
    m_startidx[etrue] = startidx;
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
  for (size_t etrue = 0; etrue < insize; ++etrue) {
     /* get the cache line */
    double *cache = &m_rescache[m_cacheidx[etrue]];
    int startidx = m_startidx[etrue];
     /* get number of valid cache entries for corresponding Etrue  */
    int cnt = m_cacheidx[etrue+1] - m_cacheidx[etrue];
    for(int off = 0; off < cnt; ++off) {
      /* current position in cache */
      size_t erec = startidx + off;
      if (erec == etrue) {
        continue;
     }
      double rEvents = cache[off];
      double delta = rEvents*events_true[etrue];

      events_rec[erec] += delta;
      int inv = 2*etrue - erec;
      /* if get outside the bins of histo then events are LOST */
      if (inv < 0 || (size_t)inv >= outsize) {
        events_rec[etrue] -= delta;
        loss += delta;
      }
      events_rec[etrue] -= delta;
    }
  }
}

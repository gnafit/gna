#include <vector>
#include <string>

#include "GNAObject.hh"

class ReactorNormAbsolute: public GNAObject,
                           public TransformationBind<ReactorNormAbsolute> {
public:
  ReactorNormAbsolute(const std::vector<std::string> &isonames);
protected:
  variable<double> m_norm;
};

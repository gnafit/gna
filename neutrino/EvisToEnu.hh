#ifndef EVISTOENU_H
#define EVISTOENU_H

#include <vector>
#include <string>

#include "GNAObject.hh"

class ReactorNormAbsolute: public GNAObject,
                           public Transformation<ReactorNormAbsolute> {
public:
  ReactorNormAbsolute(const std::vector<std::string> &isonames);
protected:
  variable<double> m_norm;
};
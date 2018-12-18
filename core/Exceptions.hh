#pragma once

#include <stdexcept>
#include <string>

#include <boost/lexical_cast.hpp>
#include <utility>

class KeyError: public std::exception {
public:
  KeyError(std::string key, std::string object = "")
    : key(std::move(key)), object(std::move(object)) { }

  const char *what() const noexcept override { return key.c_str(); }

  std::string key;
  std::string object;
};

class IndexError: public std::exception {
public:
  IndexError(int index, std::string object = "")
    : index(index), object(std::move(object)) {
    indexstr = boost::lexical_cast<std::string>(index);
  }

  const char *what() const noexcept override { return indexstr.c_str(); }

  int index;
  std::string indexstr;
  std::string object;
};

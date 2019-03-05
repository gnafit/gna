#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <stdexcept>
#include <string>

#include <boost/lexical_cast.hpp>

class KeyError: public std::exception {
public:
  KeyError(const std::string &key, const std::string &object = "")
    : key(key), object(object) { }

  const char *what() const throw() { return key.c_str(); }

  std::string key;
  std::string object;
};

class IndexError: public std::exception {
public:
  IndexError(int index, const std::string &object = "")
    : index(index), object(object) {
    indexstr = boost::lexical_cast<std::string>(index);
  }

  const char *what() const throw() { return indexstr.c_str(); }

  int index;
  std::string indexstr;
  std::string object;
};

#endif // EXCEPTIONS_H

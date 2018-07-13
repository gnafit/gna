#pragma once

#include "Args.hh"
#include "Rets.hh"

namespace TransformationTypes
{
  class Entry;
  struct FunctionArgs {
    FunctionArgs(Entry* e) : args(e), rets(e) {  }

    Args args;
    Rets rets;
  };
}


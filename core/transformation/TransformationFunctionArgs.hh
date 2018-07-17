#pragma once

#include "Args.hh"
#include "Rets.hh"
#include "Atypes.hh"
#include "Rtypes.hh"
#include "Itypes.hh"

namespace TransformationTypes
{
  class Entry;
  struct FunctionArgs {
    FunctionArgs(Entry* e) : args(e), rets(e) {  }

    Args args;
    Rets rets;
  };

  struct TypesFunctionArgs {
    TypesFunctionArgs(Entry* e) : args(e), rets(e) {  }

    Atypes args;
    Rtypes rets;
    Itypes ints;
  };
}


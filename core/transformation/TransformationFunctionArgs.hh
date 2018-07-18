#pragma once

#include "Args.hh"
#include "Rets.hh"
#include "Ints.hh"
#include "Atypes.hh"
#include "Rtypes.hh"
#include "Itypes.hh"

namespace TransformationTypes
{
  class Entry;
  struct FunctionArgs {
    FunctionArgs(Entry* e) : args(e), rets(e), ints(e) {  }

    Args args;
    Rets rets;
    Ints ints;
  };

  struct TypesFunctionArgs {
    TypesFunctionArgs(Entry* e) : args(e), rets(e), ints(e) {  }

    Atypes args;
    Rtypes rets;
    Itypes ints;
  };
}


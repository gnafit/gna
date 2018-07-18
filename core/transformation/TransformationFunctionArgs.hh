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

  /**
   * @brief Transformation Function arguments
   *
   * @author Maxim Gonchar
   * @date 07.2018
   */
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

  struct StorageTypesFunctionArgs {
    StorageTypesFunctionArgs(TypesFunctionArgs& fargs) : args(fargs.args), rets(fargs.rets), ints(fargs.ints) {  }

    Atypes& args;
    const Rtypes& rets;
    Itypes& ints;
  };
}


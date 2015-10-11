#ifndef NEUTRINO_H
#define NEUTRINO_H

class Neutrino {
public:
  enum Kind {
    Particle = 1, Antiparticle = -1,
  };
  enum Flavor {
    Electron = 0, Muon, Tau,
  };
  Kind kind;
  Flavor flavor;

  Neutrino(const Neutrino &other): kind(other.kind), flavor(other.flavor) { }
  Neutrino(Kind t, Flavor f): kind(t), flavor(f) { }
  bool operator==(const Neutrino &other) const {
    return flavor == other.flavor && kind == other.kind;
  }
  bool operator!=(const Neutrino &other) const {
    return !(*this == other);
  }
  bool sameKind(const Neutrino &other) const {
    return kind == other.kind;
  }
  bool sameFlavor(const Neutrino &other) const {
    return flavor == other.flavor;
  }
  bool isParticle() {
    return kind == Particle;
  }
  bool isAntiparticle() {
    return kind == Antiparticle;
  }
  int leptonCharge() const {
    if (kind == Particle) {
      return 1;
    } else if (kind == Antiparticle) {
      return -1;
    }
    return 0;
  }
  static Neutrino e() { return Neutrino{Particle, Electron}; }
  static Neutrino ae() { return Neutrino{Antiparticle, Electron}; }
  static Neutrino mu() { return Neutrino{Particle, Muon}; }
  static Neutrino amu() { return Neutrino{Antiparticle, Muon}; }
  static Neutrino tau() { return Neutrino{Particle, Tau}; }
  static Neutrino atau() { return Neutrino{Antiparticle, Tau}; }
};

#endif // NEUTRINO_H

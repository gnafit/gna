Introduction
======================

The purpose of this document is to briefly describe the current state
of our global neutrino analysis software and give a few of recipes
and directions towards its possible practical usage. 

The whole product is an attempt to implement the following general
principles:

* the whole structure should flexible enough to uniformly integrate
  arbitrary number of any kind of experiments into the one common flow;
* there should clean separation between analysis configuration step
  which is done once, and computations repeated a lot of times during
  fits after the configuration;
* it should be possible to modify an existing computation chain to
  transform or completely replace any of its parts (formulas, tables,
  etc) in one place without changes over the whole code base.

The way to bring these principles into action is to
introduce a number of simple independent computational blocks
representing all the inputs or mathematical operations required to
build a theoretical model of any experiment. The task of the user
(analyzer) is to use those blocks as ingredient to construct
a computational graph producing the theoretical predictions and
finally the desirable statistic. Since the blocks are *small, simple
and independent*, they may be easily implemented in a relatively
low-level language (namely C++) making all the repeating computations
fast, while all the relations between them may be expressed by means
of a slower but dynamic language (namely Python), leading to great
flexibility. Since the whole computation flow may be traced before the
computations start, it is possible to group the same computations with
different inputs into one vectorized procedure. The block structure 
also makes possible to track with high granularity the changes of
computations depending on variable inputs, potentially avoiding
useless recomputations during the fits. The drawback of the approach
is more overhead due to dynamic nature of computations structure and
keeping the parts independent, which can potentially overwhelm the
possible advantages, depending on the implementation.

In the current stage the described concept is implemented into more or
less working code. It's worth noting though, that although, as I will
try to show in the further parts of the doc, it's possible to
obtain some meaningful numbers with the current software and in spite
of the time spent to its development, I personally tend to consider it
no more than a proof-of-concept prototype requiring major adjustments,
additions, rewrites or even redesigns on all levels before any kind of
practical usage. As (as at the moment) only developer and maintainer
of the code, I consider the whole design and implementation rather
failure and strongly suggest any future developer to reimplement,
redesign any of its part or completely drop it in favor of something
more thought out without any hesitations.

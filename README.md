# Conventions
1. Matsubara frequencies with index $n\in -N,...,N-1$
   * bosonic frequencies $\omega_n = 2 n \pi T$
   * fermionic frequencies $\nu_n = (2n+1) \pi T$
   in r-channel convention the frequencies read $(\omega, \nu, \nu')$ with \omega=bosonic transferfrequency and $\nu(')=$fermionic frequencies.
2. operators in correlators:
   * 2p/4p: arbitrary order
   * 3p: in $G[\vec{O}]$ we always have $\vec{O} = (Q, F, F)$, i.e., first operator is an (bosonic) auxiliary operator, the other two are regular (fermionic) operators $F\in\{c,c^\dagger\}$.
3. abbreviations:
   * IE/sIE/aSI: ((a-)symmetric) improved estimator
   * TD: Tucker decomposition
   * [Lihm2024]: "Symmetric improved estimators for multipoint vertex functions", doi: 10.1103/PhysRevB.109.125138

# Other implicit assumptions


# Open TODOs:
* Precompilation
* Implement blockwise evaluation of FullCorrelator_KF.
* Implement the computation of IE's in the MPS language.
* Currently we always use one of the aIE for the selfenergy. However, it is argued in [Lihm2024] that incoming and outgoing lines should use different versions of the aIE.
      => Implement and test the use of different aIE for the selfenergies.
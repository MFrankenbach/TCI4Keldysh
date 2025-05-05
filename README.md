# TCI4Keldysh
This code computes imaginary- and real-frequency 4-point vertices from the multipoint numerical renormalization group (mpNRG) in quantics tensor train (QTT) format.
The vertices can be obtained in their 'full' form or decomposed into a 3d core and lower-dimensional asymptotic contributions.
Further functionalities include the computation of four-point vertices on dense, possibly nonlinear, grids and the computation of 2-4-point correlators, either
in QTT format or on a dense, linear grid.

## Getting Started
For user instructions and further details, compile the mini-manual under 'docs/manual.tex'.
Note, however, that you will need correctly formatted partial spectral functions (.mat files) to use this code.
These are normally provided by the mpNRG code by Lee et. al. [Lee2021].

## Conventions
1. Matsubara frequencies with index $n\in -N,...,N-1$
   * bosonic frequencies $\omega_n = 2 n \pi T$
   * fermionic frequencies $\nu_n = (2n+1) \pi T$
   in r-channel convention the frequencies read $(\omega, \nu, \nu')$ with \omega=bosonic transferfrequency and $\nu(')=$fermionic frequencies.
2. operators in correlators:
   * 2p/4p: arbitrary order
   * 3p: in $G[\vec{O}]$ we always have $\vec{O} = (Q, F, F)$, i.e., first operator is a (bosonic) auxiliary operator, the other two are regular (fermionic) operators $F\in\{c,c^\dagger\}$.
3. abbreviations:
   * IE/sIE/aSI: ((a-)symmetric) improved estimator
   * TD: Tucker decomposition
   * 2p,3p,4p: p=point
   * GF: correlator/Green's function
   * a,p,t: channels, i.e., different kinds of frequency parametrizations of the vertex
   * K1/2: asymptotic contributions to the vertex
   * $\Gamma$: four-point vertex

## References
1. [Lihm2024]: "Symmetric improved estimators for multipoint vertex functions", doi:10.1103/PhysRevB.109.125138
2. [Kugler2021]: "Multipoint Correlation Functions: Spectral Representation and Numerical Evaluation", doi: 10.1103/PhysRevX.11.041006
3. [Lee2021]: "Computing Local Multipoint Correlators Using the Numerical Renormalization Group", doi:10.1103/PhysRevX.11.041007
4. [Fernandez2025]: "Learning tensor networks with tensor cross interpolation: New algorithms and libraries", doi:10.21468/SciPostPhys.18.3.104

## List of contributors
Markus Frankenbach<br>
Anxiang Ge
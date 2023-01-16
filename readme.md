# OBS tilt/compliance removal

## Introduction
This is a package to remove tilt and complaince noise from the vertical-component recording of Ocean Bottom Seismometer (OBS), based on [Bell et al. (2005)](https://doi.org/10.1785/012014005), [Ye & Ritzwoller (2017)](https://doi.org/10.1093/gji/ggv024) and [Janiszewski et al. (2019)](https://doi.org/10.1093/gji/ggz051).

## Usage
You can import and use the four functions after removing response and resampling (if necesaary.)
calSpectra -> cleanSpectra -> calTransfer -> deCouple

## Algorithm

Suppose two components Z and A are linear coupled, the raw recordings are

fZ and fA in frequency domain. The part comes from A can be estimated by 

the time mean of correlation in frequency from many time windows. The Z 

recording with A part removed is:

$$f_{Z,A} = f_{Z} - \frac{ \overline{f_{Z} \cdot f_{A}^{\ast}}}{\overline{f_{A} \cdot f_{A}^{\ast}}} f_{A}$$

here the overline means time average, and $f_{Z,A}$,$f_{Z}$ and $f_{A}$ are recordings for single time window. The $\frac{ \overline{f_{Z} \cdot f_{A}^\ast}}{\overline{f_{A} \cdot f_{A}^\ast}}$ is also called transfer function, here I denote it as $TF(Z,A)$.



The same idea can be used in system with more component, like it is possible to remove A from B and Z first, and then remove B from Z:

$$f_{Z,BA} = f_{Z,A} - \frac{ \overline{f_{Z,A} \cdot f_{B,A}^\ast}}{\overline{f_{B,A} \cdot f_{B,A}^\ast}} f_{B,A}$$

Similar, I denote transfer function here as $TF(Z,B,A)$, and after some derivation, we can have:

$$TF(Z,B,A) = \frac{TF(Z,B)-TF(Z,A)TF(A,B)}{1-TF(A,B)TF(B,A)}$$

And also for more complex case:

$$TF(Z,C,B,A) = \frac{TF(Z,C,A)-TF(Z,B,A)TF(B,C,A)}{1-TF(C,B,A)TF(B,C,A)}$$



It is clear that the decouple result like $f_{Z,C,B,A}$, can be represent as linear combination of $f_{Z},f_{C},f_{B},f_{A}$ at each frequcy. It will be more convinient to save these coefficients instead of transfer function for decoupling, which was the basis function in this module.



PS:

If you consider freq band window T(Z,A), like $[f_{min},f_{cutoff}]$ for complaince removal

$$f_{Z,A} = f_{Z} - T(Z,A)\times TF(Z,A)\times f_{A}$$

the recursion relation should be:

$$TF(Z,B,A) = \frac{TF(Z,B)-[T(Z,A)+T(A,B)-T(Z,A) T(A,B)]TF(Z,A)TF(A,B)}{1-[2T(A,B)-T^2(A,B)]TF(A,B)TF(B,A)}$$

And also for more complex case:

$$TF(Z,C,B,A) = \frac{TF(Z,C,A)-[T(Z,B)+T(B,C)-T(Z,B) T(B,C)]TF(Z,B,A)TF(B,C,A)}{1-[2T(B,C)-T^2(B,C)]TF(C,B,A)TF(B,C,A)}$$


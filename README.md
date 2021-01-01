# Automatic_differentiation_for_material_modelling-dealii
Functions and guide to use automatic differentiation for material modelling in deal.II on the element and QP level

## In a nutshell
In this repository, we will outline the use of automatic differentiation (AD) in deal.II for the use in material modelling. Firstly, the already in deal.II avaiable functionality is outlined and then combined with additional features such as the use of automatic differentiation also inside highly nonlinear material models that require local (nested) iterations themself.

## What's already there
- Data types: Deal.II offers a variety of automatic differentiation (AD) packages, where herein we right now use "sacado_dfad". A comparison of different AD types can be found [here](Integration and application of symbolic and automatic differentiation frameworks in dealii_Pelteret.pdf).
- Furthermore, there are "setup" and "extractor" functions for the use of AD for the linearisation of the residual.
- And an "ad_helper" that organises the differentiation on the QP level.

@todo Add clarifications on residual and QP-level (at best some graphics)

## What's additional
A material model typically receives some type of strain and return a stress, in addition to the corresponding stress-strain tangent (and possibly further tangents). For a classical Neo-Hooke material these relations are rather straightforward and might be summarised in two lines

@todo Cauchy stress equation and Lagrangian tangent with nomenclature

(if that is already what you're looking for, jump ahead to example 1 in the implementation section).

(In case, you know all there is to know about classical linear plasticity, then just fly other the equations to get a feeling for the used notation and continue below at "Stress-strain tangent", which is essential to understand the required steps in the AD algorithm).

However, for instance, in case of plasticity that gets a bit more involved, because the relations are iterative themself (by "iterative" I mean require iterations). This means that for (small strain) linear plasticity we can write the stress as

@todo add stress equation T = kappa * ... with nomenclature

assuming that plasticity evolves and the current step is truely plastic. That still look feasible, but when you look closely you see that even though we now the newest strains eps_(n+1) from the global Newton-Raphson solution scheme,  we don't know what the new plastic strains eps_p_(n+1) are. But there must be an equation for this, right?, as there always is. Of course, here it is

@todo add discretised evolution equation for eps_p

which is simply said (when we throw all reasonable continuum mechanics conventions over board) the evolution equation for plasticity. Now we only replaced the unknown by a Lagrange multiplier herein named \gamma. To determine this unknown we use the yield function that adds another constraint (the yield condition) to our problem, namely

@todo yield function

meaning that plasticity must evolve such that the yield condition is satisfied. If we continue from our purely numerical perspective (further abusing continuum mechanics), we see that we must find the Lagrange multiplier \gamma such that the stress norm ||T^dev|| equals the current yield stress which depends on the initial yield stress and the hardening. Here, the stress as well as the hardening depend on the Lagrange multiplier, where the stress decreases and the hardening increases with increasing Lagrange multiplier. Solving the yield condition Phi=0 can be done using another local Newton-Raphson scheme that can be written as

@todo Phi_k+1 = ... = 0

where we found our first derivative, namely d_Phi_d_gamma, which we name local derivative, that will later be solved via AD. This gives us the increment in the Lagrange multiplier resulting in the total 

@todo \gamma_k+1 = ...

In case the yield function is nonlinear we repeat that computation a few times until we found the Lagrange multiplier \gamma_(k+1) that gives us a zero yield function (satisfied yield condition). When we assume linear plasticity as above, we would be done after one iteration. Anyhow we get the following results

@todo \gamma_k+1 = ..., Phi_k+1 =; eps_p_n+1=...

With these we can compute our stress as

@todo T_k+1 = ...

and go home.  Wait ...

### Stress-strain tangent (consistent tangent operator...)
..., we missed something, The stress-strain tangent

@todo C = dT_deps

which gives a slightly longer relation when using analytical differentiation

@todo C equation

However, it is important to note the used differentiation strategies and assumptions to be able to reproduce this exact results with AD.

@todo derivation, check first whether we can show the key aspect

### The key aspects to extent material models for the use of AD

@todo add and explain them

## "Levels" of AD
(That is my personal convention, if you don't like it, I'm sorry. We accept only better ideas in the "Issues".)
- 0: no AD (analytical tangents for residual linearisation and material model)
- +1: AD on QP level only
- +2: AD on cell and QP level
- -1: AD only on cell level

## Implementation of AD on QP-level
several examples on the usage starting with Neo-Hooke, plasticity, dual-surface, ...

## Implementation of AD on cell level
(a teaser on what deal.II gives you with tutorial step 72: https://github.com/dealii/dealii/pull/10393)

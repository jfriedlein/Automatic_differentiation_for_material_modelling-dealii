# Automatic_differentiation_for_material_modelling-dealii
Functions and guide to use automatic differentiation for material modelling in deal.II on the element and QP level

## In a nutshell
In this repository, we will outline the use of automatic differentiation (AD) in deal.II for the use in material modelling. Firstly, the already in deal.II avaiable functionality is outlined and then combined with additional features such as the use of automatic differentiation also inside highly nonlinear material models that require local (nested) iterations themself. The code shall be set up such that we can easily switch from AD to analytical tangents using the exact same code. So, when you decide to derive some of the tangens by hand, you can just add them to your existing code and get some speed-up.

## What's already there
- Data types: Deal.II offers a variety of automatic differentiation (AD) packages, where herein we right now use "sacado_dfad". A comparison of different AD types can be found [here](Integration and application of symbolic and automatic differentiation frameworks in dealii_Pelteret.pdf).
- Furthermore, there are "setup" and "extractor" functions for the use of AD for the linearisation of the residual.
- And an "ad_helper" that organises the differentiation on the QP level.

@todo Add clarifications on residual and QP-level (at best some graphics)

## What's additional
A material model typically receives some type of strain and returns a stress, in addition to the corresponding stress-strain tangent (and possibly further tangents). For a classical Neo-Hooke material these relations are rather straightforward and might be summarised in two lines

<img src="https://github.com/jfriedlein/Automatic_differentiation_for_material_modelling-dealii/blob/main/images/Neo-Hooke%20-%20stress%20and%20tangent.png" width="500">

(if that is already what you're looking for, jump ahead to example 1 in the implementation section).

(In case, you know all there is to know about classical isotropic plasticity, then just fly other the equations to get a feeling for the used notation and continue below at "Stress-strain tangent", which is essential to understand the required steps in the AD algorithm).

However, for instance, in case of plasticity that gets a bit more involved, because the relations are iterative themself (by "iterative" I mean require iterations). This means that for (small strain) nonlinear plasticity we can write the stress as

<img src="https://github.com/jfriedlein/Automatic_differentiation_for_material_modelling-dealii/blob/main/images/Plasticity%20-%20stress%20equation.png" width="500">

assuming that plasticity evolves and the current step is truely plastic. That still look feasible, but when you look closely you see that even though we now the newest strains eps_(n+1) from the global Newton-Raphson solution scheme,  we don't know what the new plastic strains eps_p_(n+1) are. But there must be an equation for this, right?, as there always is. Of course, here it is

<img src="https://github.com/jfriedlein/Automatic_differentiation_for_material_modelling-dealii/blob/main/images/Plasticity%20-%20plastic%20strain%20evolution.png" width="500">

which is simply said (when we throw all reasonable continuum mechanics conventions over board) the evolution equation for plasticity. Now we only replaced the unknown by a Lagrange multiplier herein named \gamma. To determine this unknown we use the yield function that adds another constraint (the yield condition) to our problem, namely

@todo yield function

meaning that plasticity must evolve such that the yield condition is satisfied. If we continue from our purely numerical perspective (further abusing continuum mechanics), we see that we must find the Lagrange multiplier \gamma such that the stress norm ||T^dev|| equals the current yield stress which depends on the initial yield stress and the hardening. Here, the stress as well as the hardening depend on the Lagrange multiplier, where the stress decreases and the hardening increases with increasing Lagrange multiplier. Solving the yield condition Phi=0 can be done using another local Newton-Raphson scheme that can be written as

@todo Phi_k+1 = ... = 0

where we found our first derivative, namely d_Phi_d_gamma, which we name local derivative, that will later be solved via AD. This gives us the increment in the Lagrange multiplier resulting in the total 

@todo \gamma_k+1 = ...

In case the yield function is nonlinear we repeat that computation a few times until we found the Lagrange multiplier \gamma_(k+1) that gives us a zero yield function (satisfied yield condition). When we assume linear plasticity, we would be done after one iteration. Moreover, linear plasticity would give a constant local derivative and simply several aspects of the general algorithm for nonlinear plasticity. Anyhow, we get the following results

@todo \gamma_k+1 = ..., Phi_k+1 =; eps_p_n+1=...

With these we can compute our stress as

@todo T_k+1 = ...

and go home.  Wait ...

### Stress-strain tangent (consistent elastoplastic tangent modulus)
..., we missed something, The stress-strain tangent

@todo C = dT_deps

which gives a slightly longer relation when using analytical differentiation

@todo C equation and derivations with equation numbers

However, it is important to note the used differentiation strategies and assumptions to be able to reproduce this exact result with AD.
In above eq. (XXX), we compute the derivative of the Lagrange multiplier with respect to the strains as

@todo add again the d_gamma_d_eps derivative.

(Now some rather long monologues follow, that try to describe the issues you will face. Please, take the time to read it, it will save you some trouble. In case you prefer code, you can also first jump into the code, where the issues are briefly repeated and directly solved, but then I recommend coming back to this text.)

We neglect the second term, because by definition the final yield function Phi_(k+1) will be practically zero from the previous iterations and we will only compute the stress-strain derivative when we found the correct Lagrange multiplier that satisfies the yield condition. If you like second derivatives (AD doesn't like them and would require a fully two level derivative scheme ... just expensive), you could derive the term and avoid this approximation. When we focus on the first term, we see that the derivative needs to be evaluated at the newest solution (k+1), because that is the solution, which we linearise around (see figure Phi-gamma).

@todo add figure with Phi-gamma and linearisation, k and k+1

When we use the analytical tangent from above, we exactly have to do this. So, we would compute the d_Phi_d_gamma derivative again for the newest values and simply incorporate that in above eq. (xxx). For AD we also need this newest local tangent, but the incorporation is a bit more involved. For AD we extract the derivative from the stress, so we need to introduce the newest local derivative into this variable. However, there is no explicit way to introduce that. This can be solved by computing another iteration, which computes a new Lagrange multiplier (being practically the same as the gamma_k solution, so we do no harm) that is based on the newest d_Phi_d_gamma. Now we have the newest local derivative, but the stress variable still knows nothing about this, so we update the history and compute the stress again that now contains the new information. This additional iteration is called "AD_postStep" in the following and is  key aspect #3. 

## The key aspects to extent material models for the use of AD
We already outlined #3, so now we summarise the remaining steps:
* #1 Initialise values AND derivatives: 
You will  stumble other this in the first local iteration to find the Lagrange multiplier, when you want to extract the local derivative d_Phi_d_gamma. In the following, the important equations of the first iteration (k=0) are summarised for a classical analytical model

@todo add k=0: init value, Tk ... d_Phi_d_gamma

If we use this algorithm for automatic differentiation, we would get an undefined derivative d_Phi_d_gamma. Why? Let's go through the derivatives that AD would see. First, we assign the initial value of the plastic strains to the value of the history eps_p_n. Note that the history contains only values, but no derivatives (as it should be), since it is only "a point". Furthermore, we initialise the value of the Lagrange multiplier as zero, where its derivative with respect to the Lagrange multiplier (itself) is correctly set as 1, by the AD setup (will be shown in the code later). Next we compute the (trial) stress and the yield function. When you look at the equations you see that the yield function never involved the Lagrange multiplier, simply because for the analytical case we would not need it, because its value is zero anyways. To sum up, what AD sees in this yield function is its value, but because the Lagrange multiplier was not used to obtain the value (from what AD can see) it assumes that the yield function does not depend on the Lagrange multiplier and thus cannot give you the derivative with respect to the Lagrange multiplier. And that is exactly the probem. Only in the next iteration, when we computed the Lagrange multiplier from the current yield function and updated the history, which alters the yield function, would we get a derivative.

What we can do about it: The concept to remember is "Always initialise a variable's value AND its derivative"! Okay, that sounds nice, but how can I initialise a derivative? For AD this is rather simple, but AD extracts the derivatives from the equations themself. So, instead of intialising e.g. the plastic strain only by the old value we do

@todo add eps_p_k = eps_p_n + gamma * n

And yes, in the first iteration the value of gamma is zero, so we change absolutely nothing in the values. BUT, we introduce the dependency, thus also the derivative, of eps_p_k on the Lagrange multiplier and that is exactly what we want. You will see the consequence this has on the algorithm later in the code.

@todo add note on everything needs to be properly initialised (e.g. also the evolution direction)

* #2 Reset the nested derivative
The next issue occurs in each iteration and will completely destroy your convergence if not considered. Again, we look at an excerpt of the algorithm

@todo part 2 with Phi and derivative and new derivative

The first line therein just states that we compute the yield function from the Lagrange multiplier (a fact). Next, we see how the new Lagrange multiplier gamma_(k+1) is computed from the old one and the update that uses the local derivative d_Phi_d_gamma. When we look at this line as it is implemented we see
```
gamma_k = gamma_k - Phi_k / d_Phi_d_gamma;
```
where we got the local derivative (just a scalar value) from AD as you will see in the full code later.  We have already seen that for AD we operate on the values of variables and on the derivatives. The gamma_k on the right-hand side of the equation is our AD dof, which correctly contains the derivative with respect to itself as 1. Phi_k also depends on the Lagrange multiplier, so it also contains some derivative with respect to gamma as it should, also in the first iteration (see #1). d_Phi_d_gamma is just a scalar and thus only scales the value of Phi_k and the therein stored derivatives. The difference of these two terms is the difference of their values, where the operations also acts on their derivatives. Now, we can be sure that the difference of 1 (the derivative of gamma_k wrt itself) and the derivative stored in Phi_k wrt gamma will not be 1, because the second term cannot be zero (else d_Phi_d_gamma would be zero and give a division by zero). Thus, we derivative we write into gamma_k wrt to gamma is different from 1. Okay, what do we have so far. We updated the Lagrange multiplier to its new value and changed the related derivatives. The updated gamma_k has a new value and the derivative that is stored inside and describes the relation to the Lagrange multiplier gamma is different from 1. But wait ... that would mean that the derivative of the gamma_k (being the Lagrange multiplier) with respect to gamma (being the Lagrange multiplier) is not 1, but the derivative of a variable wrt itself has to be 1. This problem arises from the nested iterations, so we update a variable thus overwriting its value and derivative to use its new "look" in the next iteration. However, in the next iteration we want to take the derivative of the yield function wrt the new gamma not the old one. So if we were not using a loop, we would write
```
gamma_k1 = gamma_k - ...;
```
and then take the derivatives wrt gamma_k1. But because we are lazy and want to keep using the name gamma_k, we have to tell AD that from now on, derivatives wrt the Lagrange multiplier shall be performed wrt to the new gamma_k. All of this sounds very complicated, but in the end it boils down to setting the derivative in gamma_k wrt gamma to 1 manually, which is enough to tell AD that the new gamma_k is the one we want.

@todo the code piece on the resetting

* #3 Additional AD post step

As already outlined in the section on the stress-strain tangent above, we run through an additional iteration after we found the converged state to update the derivatives inside the stress.

* #4 Points of non-differentiability

That's an issue you will encounter regularly. In the beginning, your algorithm will fail often due to variables that contain 'nan' or 'inf' in their values and/or derivatives. AD is here like a geologist, it will dig out all the points of non-differentiability and show them to you. Isn't that nice? No, it is super annoying. Enough talking, let's get to the core. You now it from e.g. a hyperbola y=1/x that gets 'inf' when you compute it with x=0. Now, AD also computes its derivative at the given point let's say (x=0) as y'=-1/x², which is also 'inf' for x=0 (To all mathematicians, I apologize for the sloppy use of infinity). In case, you anyway introduce some small deviation e.g. as 1/(x+1e-8) to avoid this or do some extrapolation for small values, this would also resolve the undefined tangent. The nasty thing about AD, however, is that AD will find points of non-differentiability where the 'inf' issue only occurs for the derivative, but not for the equation (e.g. y=...) itself. But because you only implement the equation (y=...) and AD determines the derivative in the background (that you never see as an equation y'), you often don't see the issue beforehand.

Let's play a game. I list some expressions and functions and you yell 'yes', when you found one that exhibits points of non-differentiability. 'x' is our independent variable as a scalar. 'X' is our independent variable as a tensor. 'n' is just an arbitrary scalar. 'y' is the result and we want AD to compute y'=dy/dx:
* y=sqrt(x)
* y=X.norm()
* y=ln(x)
* y=x^(2/3)
* y=1/x
* @todo add many more

So, go over the above list and see whether you find all the "problematic" candidates, describing the ones where we need to do some extra precautions to avoid our geologist (AD) to dig out dirt (Don't get me wrong, AD and geology are beauties, but there is some dirt involved in both of them). And ... have you noticed it? Each of the above expressions is non-differentiable at some point. The above list is a summary of functions you have to look out for and treat accordingly. The 'norm()' might be one of the nasties, I personally stumbled over so often. In essence, it also contains a square root, whose derivative is '0.5*x^(-0.5)', also part of the '1/x' family, but admittingly well hidden. We will show some ways to work around these problems in the code (in a nutshell: either catch it by 'if ...', add a perturbation like '+1e-8' or extrapolate the expression close to the critical value).


## Implementation of AD on QP-level
Now we talk enough about the theory, lets dive into the code and see.

several examples on the usage starting with Neo-Hooke, plasticity, dual-surface, ...


## "Levels" of AD
(That is my personal convention, if you don't like it, I'm sorry. We only accept better ideas in the "Issues" section, not complaints.)
- 0: no AD (analytical tangents for residual linearisation and material model)
- +1: AD on QP level only
- +2: AD on cell and QP level
- -2: AD only on cell level

Why all that? To be able to easily check what kind of differentiation to use on QP and cell level:
* if (AD_level==0) -> analytical tangents everywhere
* if (AD_level>0) -> use AD on QP level
* if (abs(AD_level)==2) -> use AD on cell level

which captures the four cases above quite nicely.

## Implementation of AD on cell level
(a teaser on what deal.II gives you with tutorial step 72: https://github.com/dealii/dealii/pull/10393)


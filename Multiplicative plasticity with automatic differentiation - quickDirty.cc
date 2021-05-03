// Sacado
#include <Sacado.hpp>
#include "../Sacado_Tensors_dealII/Sacado_Wrapper.h"
#include "../Sacado_Tensors_dealII/Sacado-auxiliary_functions.h"
#include <deal.II/differentiation/ad/ad_helpers.h>

// Storage of the tangents
#include "Tangent_groups.h"


 /*
  * multiplicative plasticity
  */

 template<typename Number>
 void elastoplasticity_multi
 					(
 						 /*input->*/ const Tensor<2,3,Number> &F, std::shared_ptr<PointHistory<dim> > lqph_QPk,
 						 /*output->*/ SymmetricTensor<2,3,Number> &stress_S, Tangent_groups_u2<dim> &Tangents,
 						 bool &GG_mode_requested
 					);

 void prepare_AD_helper_fstrain_plasti__ST_multi
	(
		Tensor<2,3> &F, std::shared_ptr<PointHistory<dim> > lqph_QPk,
		SymmetricTensor<2,3> &stress_S, Tangent_groups_u2<dim> &Tangents,
		bool &GG_mode_requested
	);


// In der assembly routine lässt sich das dann z. B. als "prepare_AD_helper_fstrain_plasti__ST_multi( DefoGradient_3D, lqph[k], stress_S_3D, Tangents, GG_mode_requested );" aufrufen
"
template <int dim>
void MaterialModel<dim>::prepare_AD_helper_fstrain_plasti__ST_multi
						(
							Tensor<2,3> &F, std::shared_ptr<PointHistory<dim> > lqph_QPk,
							SymmetricTensor<2,3> &stress_S, Tangent_groups_u2<dim> &Tangents,
							bool &GG_mode_requested
						)
{
	// Independent variables
	 // @todo not the place I would want this to be, it should be handled inside the matmod depending on whether we use local indep. variables or not
	 const unsigned int n_independent_parameters = 1; // Here we don't count the components
	 std::vector<unsigned int> AD_dofs_storage_n_dofs (n_independent_parameters);
	 AD_dofs_storage_n_dofs[enums::defoGrad_F] = Tensor<2,3>::n_independent_components;

	 std::vector<unsigned int> AD_dofs_storage_pos (n_independent_parameters,0);

	 // Store the AD_dofs in the given order (set via the enumerators)
	 unsigned int summed_n_dofs_j = 0;
	 for ( unsigned int i=1; i<n_independent_parameters; i++ )
	 {
		 summed_n_dofs_j += AD_dofs_storage_n_dofs[i-1];
		 AD_dofs_storage_pos[i] = summed_n_dofs_j;
	 }
	 // The number of independent variables is computed from the highest storage position plus the
	 // length of this last independent variable. (More intuitively we could also just add up all the n_dofs for each variable.)
	  const unsigned int n_independent_variables = AD_dofs_storage_pos.back() + AD_dofs_storage_n_dofs.back();

	 const FEValuesExtractors::Tensor<2> F_dofs (AD_dofs_storage_pos[enums::defoGrad_F]);

	// Dependent variables
	 const FEValuesExtractors::SymmetricTensor<2> S_dofs (0);
	 const unsigned int n_dependent_variables = SymmetricTensor<2,3>::n_independent_components;

	// AD helper
	 Differentiation::AD::VectorFunction<3,Differentiation::AD::NumberTypes::sacado_dfad,double> ad_helper (n_independent_variables, n_dependent_variables);

	// Init the independent variables and create their AD data types counterparts
	 ad_helper.register_independent_variable( F, F_dofs );

	 const Tensor<2,3,fad_double> F_AD = ad_helper.get_sensitive_variables(F_dofs);

	// Call the material mode to get the stress
	 SymmetricTensor<2,3,fad_double> S_AD;
	 elastoplasticity_multi( /*input->*/ F_AD, lqph_QPk,
					  	    /*output->*/ S_AD, Tangents, GG_mode_requested );

	// Define the stress as a dependent variable
	 ad_helper.register_dependent_variable(S_AD,S_dofs);

	// Declare variables to store the values and the jacobian/derivatives
	 Vector<double> values ( ad_helper.n_dependent_variables() );
	 FullMatrix<double> jacobian ( ad_helper.n_dependent_variables(), ad_helper.n_independent_variables() );

	// Extract the values and derivatives from the ad_helper
	 ad_helper.compute_values(values);
	 ad_helper.compute_jacobian(jacobian);

	// Store the results into the stress and tangent tensors
	 stress_S = ad_helper.extract_value_component(values,S_dofs);
	 Tensor<4,3> dS_dF;

		if ( (F-StandardTensors::I<3>()).norm() < 1e-8 )
		{
			Tensor<4,3> dc;
			 for (unsigned int i = 0; i < 3; ++i)
				 for (unsigned int j = 0; j < 3; ++j)
					 for (unsigned int k = 0; k < 3; ++k)
						 for (unsigned int l = 0; l < 3; ++l)
							 dc[i][j][k][l] = lambda * StandardTensors::I<3>()[i][j] * StandardTensors::I<3>()[k][l]
													+ 2.*mu*StandardTensors::I<3>()[i][k] * StandardTensors::I<3>()[j][l];

			 Tangents.set_dS_dF( dc );
		}
		else
		{
			dS_dF = ad_helper.extract_jacobian_component(jacobian,S_dofs,F_dofs);
			Tangents.set_dS_dF( dS_dF );
		}
}


template <int dim>
template <typename Number>
void MaterialModel<dim>::elastoplasticity_multi
					(
						 /*input->*/ const Tensor<2,3,Number> &F, std::shared_ptr<PointHistory<dim> > lqph_QPk,
						 /*output->*/ SymmetricTensor<2,3,Number> &stress_S, Tangent_groups_u2<dim> &Tangents,
						 bool &GG_mode_requested
					)
{
	SymmetricTensor<4,3> dc;
	Tensor<4,3> dc_unsym;

	if ( (F-StandardTensors::I<3>()).norm() < 1e-8 )
	{
		//stress_S remains empty
		stress_S = SymmetricTensor<2,3,Number>();

		// iCplast is not initially zero
		 lqph_QPk->eps_p_n = StandardTensors::I<3>();

		 for (unsigned int i = 0; i < 3; ++i)
			 for (unsigned int j = 0; j < 3; ++j)
				 for (unsigned int k = 0; k < 3; ++k)
					 for (unsigned int l = 0; l < 3; ++l)
						 dc_unsym[i][j][k][l] = lambda * StandardTensors::I<3>()[i][j] * StandardTensors::I<3>()[k][l]
												+ 2.*mu*StandardTensors::I<3>()[i][k] * StandardTensors::I<3>()[j][l];
	}
	else
	{
		// Extract history variables
		 double alpha_n = lqph_QPk->alpha_n;
		 SymmetricTensor<2,3> iRCG_p_n = lqph_QPk->eps_p_n;

		// inverse defoGrad
		 const Tensor<2,3,Number> F_inv = invert(F);
		 const Number detF = determinant(F);

		// elastic trial finger tensor
		// @todo Check the proper contraction also below
		 SymmetricTensor<2,3,Number> b_e_trial = symmetrize( F * Tensor<2,3> (iRCG_p_n) * transpose(F) );

		// spectral decomposition of elastic trial finger tensor
		 std::vector<Number> eigenvalues_sqrt (3);
		 std::vector< Tensor<1,3,Number> > eigenvector (3);
		 for (unsigned int i = 0; i < 3; ++i)
		 {
			eigenvalues_sqrt[i] = std::sqrt( eigenvectors(b_e_trial)[i].first );
			eigenvector[i] = eigenvectors(b_e_trial)[i].second;
		 }

		// trial principal deviatoric Kirchhoff stress
		 SymmetricTensor<2,3,Number> taudev_trial;

		// Write only the principal values on the diagonal (i,i)
		for (unsigned int i = 0; i < 3; ++i)
			taudev_trial[i][i] = 2.*mu*std::log(eigenvalues_sqrt[i]) - 2./3. * mu * std::log(detF);

		// norm of \a taudev_trial
		 Number taudev_trial_norm = taudev_trial.norm();

		// pressure (Kirchhoff type)
		 Number p = kappa * std::log(detF) ;

		// Trial yield condition
		 Number yield_stress_current = yield_stress + K*alpha_n + yield_stress_incr * ( 1. - std::exp(-parameter.K_exp*alpha_n) );
		 Number Phi_trial = taudev_trial_norm - std::sqrt(2./3.) * yield_stress_current;

		 Number dPhi_dlambda = 2.*mu + 2./3. * ( K + yield_stress_incr * parameter.K_exp * std::exp( -parameter.K_exp * alpha_n ) );
		 Number delta_lambda = 0.;

		 Number alpha_n_k = alpha_n;

		if ( Phi_trial > -1e-8 ) // plastic
		{
			unsigned int k=0; // declared outside of the scope for the convergence check below
			for ( ; k < max_nbr_MatMod_its; k++ )
			{
				delta_lambda += Phi_trial/dPhi_dlambda;
				alpha_n_k += sqrt(2./3.) * Phi_trial/dPhi_dlambda;
				yield_stress_current = yield_stress + K*alpha_n_k + yield_stress_incr * ( 1. - std::exp(-parameter.K_exp*alpha_n_k) );
				Phi_trial = taudev_trial_norm - std::sqrt(2./3.) * yield_stress_current - delta_lambda*2.*mu;
				if ( std::abs(Phi_trial) <= 1e-8 )
					break; // converged
				dPhi_dlambda = 2.*mu + 2./3. * ( K + yield_stress_incr * parameter.K_exp * std::exp( -parameter.K_exp * alpha_n_k ) );
			}
		}

		// direction
		 SymmetricTensor<2,3,Number> n_n1 = taudev_trial / taudev_trial_norm;

		SymmetricTensor<2,3,Number> tau_dev = taudev_trial - delta_lambda * 2. * mu * n_n1;

		// spectral composition of Kirchhoff stress
		// @todo-optimize use eigenbasis
		 SymmetricTensor<2,3,Number> tau;
		 for (unsigned int i = 0; i < 3; ++i)
			 tau += tau_dev[i][i] * symmetrize( outer_product( eigenvector[i], eigenvector[i] ));

		 tau += p*StandardTensors::I<3>();

		// update elastic stretch
		 std::vector<Number> lame (3);
		 for (unsigned int i = 0; i < 3; ++i)
			 lame[i] = std::exp( std::log(eigenvalues_sqrt[i]) - delta_lambda * n_n1[i][i] );

		// spectral composition of elastic finger tensor
		 SymmetricTensor<2,3,Number> be;
		 for (unsigned int i = 0; i < 3; ++i)
			 be += lame[i]*lame[i] * symmetrize(outer_product( eigenvector[i], eigenvector[i] ));

		// update inverse plastic Cauchy Green
		 SymmetricTensor<2,3> iRCG_p_n_k = SacadoQP::get_value( symmetrize( F_inv * Tensor<2,3,Number>(be) * transpose(F_inv) ) );

		// Cauchy stress
		 SymmetricTensor<2,3,Number> Cauchy_stress = 1./detF * tau;

		// Write the updated values into the return argument
		 lqph_QPk->update_elpl_history_tmp( iRCG_p_n_k, SacadoQP::get_value(alpha_n_k) );

		 stress_S = detF * symmetrize( F_inv * Tensor<2,3,Number>(Cauchy_stress) * transpose(F_inv));
	}

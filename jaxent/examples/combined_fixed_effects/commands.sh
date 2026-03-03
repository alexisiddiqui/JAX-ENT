# Run the cross-experiment mixed effects model analysis
python jaxent/examples/combined_fixed_effects/cross_experiment_mixed_effects_model.py \
  --exp-teaa "jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_processed__optimise_test_SIGMA_500__20260216_232705" \
  --exp-moprp-rw "jaxent/examples/2_CrossValidation/fitting/jaxENT/_processed__optimise_quick_test_FIGURE_SIGMA_500__20260217_163516" \
  --exp-moprp-rwbv "jaxent/examples/3_CrossValidationBV/fitting/jaxENT/_processed__optimise_quick_test_test_SIGMA_500_lr1.0_BV_objectve_scale1.0__20260217_165612"
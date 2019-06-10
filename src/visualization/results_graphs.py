import numpy as np

# Results per percentage labeled

# (Free ratio)

# IONOSPHERE (30 runs)

percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
roc_score_ionosphere = [0.8851, 0.8810, 0.8749, 0.8480, 0.8394, 0.8395, 0.8487, 0.8120, 0.8229, 0.8185]
p_value_ionosphere = [5.33*np.power(10, -5),  4.64*np.power(10, -5), 0.23*np.power(10, -5), 0.21*np.power(10, -9),
                      2.69*np.power(10, -5), 4.51*np.power(10, -6), 2.28*np.power(10, -7), 4.07*np.power(10, -12),
                      2.25*np.power(10, -12), 2.51*np.power(10, -11)]
labeled_ionosphere = [17, 27, 40, 57, 68, 82, 100, 97, 104, 119, 120]

# DIABETES (30 runs)

percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
roc_score_diabetes = [0.7790, 0.7773, 0.7830, 0.7843, 0.7762, 0.7734, 0.7727, 0.7683, 0.7676, 0.7705]
p_value_diabetes = [4.92*np.power(10, -8),  4.16*np.power(10, -8), 2.05*np.power(10, -14), 1.8*np.power(10, -16),
                    6.20*np.power(10, -5), 0.1282, 0.4697, 6.03*np.power(10, 4), 1.50*np.power(10, -4), 0.4112]
labeled_diabetes = [40, 80, 110, 148, 185, 220, 250, 275, 285, 285]

# (Fixed ratio)

# IONOSPHERE (30 runs)

percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
roc_score_ionosphere = [ _, _, _, _, _, _, _, _, _, _]
p_value_ionosphere = [_*np.power(10, _),  _*np.power(10, _), _*np.power(10, _), _*np.power(10, _), _*np.power(10, _), _*np.power(10, _), _*np.power(10, _)
, _*np.power(10, _), _*np.power(10, _), _*np.power(10, _)]
labeled_ionosphere = [_, _, _, _, _, _, _, _, _, _, _]

# DIABETES (30 runs)

percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
roc_score_diabetes = [0.7793, 0.7762, 0.7838, 0.7847, 0.7757, 0.7744, 0.7734, 0.7672, 0.7658, 0.7692]
p_value_diabetes = [5.60*np.power(10, -7),  2.38*np.power(10, -6), 1.33*np.power(10, -16), 8.40*np.power(10, -14),
                    8.75*np.power(10, -5), 1.16*np.power(10, -3), 9.6*np.power(10, -2), 6.76*np.power(10, -4),
                    5.39*np.power(10, -6), 1.2*np.power(10, -2)]
labeled_diabetes = [40, 78, 108, 145, 185, 190, 210, 234, 253, 268, 268]
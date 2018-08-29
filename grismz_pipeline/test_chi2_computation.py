from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

# Define all arrays
# grism and obs photometry arrays
grism_lam_obs = np.linspace(6000, 9500, 88)
grism_flam_obs = np.arange(88)
grism_ferr_obs = np.ones(88)
phot_lam = np.array([4000, 4500, 7000, 10000, 15000])
phot_flam_obs = np.array([5, 10, 50, 30, 25])
phot_ferr_obs = np.array([0.05, 1, 2, 0.5, 0.5])

# model arrays
total_models = 10000
model_spec = np.ones((10000, 88))
all_filt_flam_model = np.ones((total_models, 5)) * 0.5

model_spec_list = []
for j in range(total_models):
    model_spec_list.append(model_spec[j].tolist())

# Insert photometry 
count = 0
for phot_wav in phot_lam:
    
    if phot_wav < grism_lam_obs[0]:
        lam_obs_idx_to_insert = 0

    elif phot_wav > grism_lam_obs[-1]:
        lam_obs_idx_to_insert = len(grism_lam_obs)

    else:
        lam_obs_idx_to_insert = np.where(grism_lam_obs > phot_wav)[0][0]
    print count, phot_wav, lam_obs_idx_to_insert

    grism_lam_obs = np.insert(grism_lam_obs, lam_obs_idx_to_insert, phot_wav)
    grism_flam_obs = np.insert(grism_flam_obs, lam_obs_idx_to_insert, phot_flam_obs[count])
    grism_ferr_obs = np.insert(grism_ferr_obs, lam_obs_idx_to_insert, phot_ferr_obs[count])

    for i in range(total_models):
        model_spec_list[i] = np.insert(model_spec_list[i], lam_obs_idx_to_insert, all_filt_flam_model[i, count])

    count += 1

# Now convert the model spectra list back to a numpy array to compute the chi2
del model_spec
model_spec = np.asarray(model_spec_list)

# Check shapes
print model_spec.shape
print grism_lam_obs.shape
print grism_flam_obs.shape
print grism_ferr_obs.shape

# Plot to check
for i in range(1):  # all the models and the data are the exact same # you only need to check one plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot grism + obs photometry
    ax.plot(grism_lam_obs, grism_flam_obs, 'o-', color='k', markersize=2)
    ax.fill_between(grism_lam_obs, grism_flam_obs + grism_ferr_obs, grism_flam_obs - grism_ferr_obs, color='lightgray')
    ax.scatter(phot_lam, phot_flam_obs, s=10, color='r', zorder=10)

    # plot model + model photometry
    ax.plot(grism_lam_obs, 25*model_spec[i], 'o-', color='b', markersize=2)  # using an alpha=25 # when I printed stuff I saw that alpha~26
    ax.scatter(phot_lam, 25*all_filt_flam_model[i], s=10, color='r', zorder=10)

    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

# Chi2 computation
alpha_ = np.sum(grism_flam_obs * model_spec / (grism_ferr_obs**2), axis=1) / np.sum(model_spec**2 / grism_ferr_obs**2, axis=1)
chi2_ = np.sum(((grism_flam_obs - (alpha_ * model_spec.T).T) / grism_ferr_obs)**2, axis=1)

print alpha_
print chi2_

print alpha_.shape
print chi2_.shape
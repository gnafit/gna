name = '2peaks_cov'
mode = 'relative'
uncorr_uncs = [1, 0.5] 
# 'keep' stands for keeping uncorrelated uncertainties of parameters and 'override' stands
# for substituting them values from uncorr_uncs
policy = 'override' 
params = ['E0', 'Width']
cov_mat = [[1, 0.7], [0.7, 1]]

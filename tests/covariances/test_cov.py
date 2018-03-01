# Name assosiated with a set of parameters and covariance matrix. Will be used
# to retrieve it 
name = 'test_cov'
mode = 'relative'
uncorr_uncs = [0.5, 0.1] 
# 'keep' stands for keeping uncorrelated uncertainties of parameters and 'override' stands
# for substituting them values from uncorr_uncs
policy = 'keep' 
params = ['Width', 'E0']
cov_mat = [[1, 0.1], [0.1, 1]]

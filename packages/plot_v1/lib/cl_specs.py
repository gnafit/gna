import re
import scipy.stats

def parselevel(pvspec: str) -> float:
    '''Trying to parse valid list of p-values from input string.
    Valid options that are recognized by parser:
    - cl -- confidence level, with value in either [50,100] or [0,1]
    - pv -- p-value, with value in either [50,100] or [0,1]
    - s, sigma -- number of sigmas from 1-D Gaussian distribution, values in [0, inf)
    '''
    try:
        n, t = re.match('([0-9.]+)(.*)', pvspec).groups()
    except AttributeError:
        msg = f"invalid CL specifier {pvspec}"
        raise ValueError(msg)
    n = float(n)
    if t in 'cl':
        if 50 < n < 100:
            pv = 1 - n/100
        elif 0 < n < 1:
            pv = 1 - n
        else:
            msg = f"invalid value for CL: {n}"
            raise ValueError(msg)
    elif t == 'pv':
        if 50 < n < 100:
            pv = 1 - n/100
        elif 0 < n < 1:
            pv = 1 - n
        else:
            msg = f"invalid value for p-value: {n}"
            raise ValueError(msg)
    elif t in ('s', 'sigma'):
        pv = 1 - scipy.stats.chi2.cdf(n**2, 1)
    else:
        msg = f"invalid label CL specifier: {t}"
        raise ValueError(msg)
    return pv

def criticalvalue(pvspec: str) -> float:
    return scipy.stats.chi2.ppf(1. - parselevel(pvspec), 1)

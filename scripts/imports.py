import matplotlib.pyplot as plt
import numpy as np
from gna.bindings import common
from mpl_tools.helpers import plot_hist 
from matplotlib.backends.backend_pdf import PdfPages
from os.path import join, abspath

detectors_gna = ['AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34']
detectors_dyboscar = ['EH1_AD1', 'EH1_AD2', 'EH2_AD1', 'EH2_AD2', 'EH3_AD1', 'EH3_AD2', 'EH3_AD3', 'EH3_AD4']
reactors  = ['DB1', 'DB2', 'LA1', 'LA2', 'LA3', 'LA4']
__old = {}

def restore_osc_prob():
    ns = __old['ns']
    ns['SinSq12'].set(__old['SinSq12'])
    ns['SinSq13'].set(__old['SinSq13'])

def set_noosc(osc_ns):
    __old['ns'] = osc_ns
    SinSq12 = osc_ns['SinSq12']
    __old['SinSq12'] = SinSq12.value()
    SinSq12.set(0.)
    SinSq13 = osc_ns['SinSq13']
    __old['SinSq13'] = SinSq13.value()
    SinSq13.set(0.)

def match(iterable, val):
    return [_ for _ in iterable if val in _]

def plot_unoscillated_spectra(env):
    for det in detectors_gna:
        env.get('gna/reactor_pred_noosc.{}'.format(det)).plot()
    plt.show()

def get_from_ROOT(path):
    import uproot as ur 
    data_file = ur.open(path)
    return data_file

def plot_ratio(gna, dyboscar, **kwargs):
    title = kwargs.pop('title', '')
    weight = kwargs.pop('weight')

    dyb_data, dyb_bins = dyboscar.numpy()
    fig, (axis_spectra, axis_ratio) = plt.subplots(ncols=1, nrows=2)
    fig.subplots_adjust(hspace=0.05)
    fig.set_tight_layout(True)

    gna.plot_hist(axis=axis_spectra, label='gna')
    plot_hist(dyb_bins, dyb_data*weight, label='dyboscar', axis=axis_spectra)
    axis_spectra.legend()
    axis_spectra.set_title("Spectra {}".format(title), fontsize=16)

    ratio = gna.data() / dyb_data*weight
    print(title + '\n', ratio)

    plot_hist(dyb_bins, ratio, axis=axis_ratio, label='ratio')
    lims = kwargs.pop('lims')
    if lims is not None:
        axis_ratio.set_ylim(lims[0], lims[1])
    axis_ratio.axhline(y=1.0, linestyle='--', color='grey', alpha=0.5)
    axis_ratio.legend()
    axis_ratio.set_title("Ratio", fontsize=16)

    printer_to_file = kwargs.get('pp', None)
    if printer_to_file:
        printer_to_file.savefig(fig)
        plt.close('all')

def plot_all(gna_template, dyboscar_template, env, root_file, output=None,
             efflivetime_weights=None, lims=None, draw_only=None):
    pp = None
    if efflivetime_weights is None:
        efflivetime_weights = np.ones(len(detectors_gna))
    if output:
        pp = PdfPages(str(abspath(output) + '.pdf'))
    for ad_gna, ad_dyb, weight in zip(detectors_gna, detectors_dyboscar, efflivetime_weights):
        if draw_only is not None:
            if ad_gna not in draw_only:
                continue
        gna_obs = env.get(gna_template.format(ad_gna))
        dyb_obs = root_file[dyboscar_template.format(ad_dyb)]
        plot_ratio(gna_obs, dyb_obs, title=ad_gna, pp=pp, weight=weight, lims=lims)
    if pp:
        pp.close()
    else:
        plt.show()


root_file = get_from_ROOT('./data/p15a/dyboscar_new/fit_scaled_all.shape_cov.ihep_spec.root')

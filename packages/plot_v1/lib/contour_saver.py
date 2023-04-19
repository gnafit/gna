from pathlib import Path
from typing import Union, Sequence, List, Tuple, Type
from collections import abc

import numpy as np
import numpy.core.records
import matplotlib.pyplot as plt
import matplotlib.contour
import matplotlib.path
import h5py

import ROOT

Contours = Type[matplotlib.contour.QuadContourSet]
MPath = Type[matplotlib.path.Path]
NList = List[np.ndarray]
Bestfit = List[Tuple[float, Union[float, Tuple[float,float]]]]

_savers = {}

def savemap(saver_name: Union[str, Sequence[str]] , *args, **kwargs):
    '''Registration decorator to enable saving contours to different storage
       formats'''
    def setupmap(cls):
        if isinstance(saver_name, abc.Sequence): 
            for name in saver_name:
                _savers[name] = cls
        else:
            _savers[saver_name] = cls
        return cls
    return setupmap

class ContourSaver:
    '''Adapter class to save contours, calls concerete
    implementation guessed from output extension'''
    __known_extensions = _savers.keys()
    def __init__(self, output: Path, contours: Contours, bestfit):
        self.output = output
        self.extension = output.name
        self.contours = contours
        self.bestfit = bestfit
        self.save_to_file(contours)

    def save_to_file(self, contours: Contours)->None:
        concere_saver = _savers[self.extension](self.contours, self.bestfit)
        concere_saver.save_to_file(self.output)

    @property
    def extension(self)->str:
        return self._extension

    @extension.setter
    def extension(self, fname: str)->None:
        for ext in ContourSaver.__known_extensions:
            if fname.endswith(ext):
                self._extension = ext
                return
        raise KeyError(f"Saving to {fname} format is unsupported")


class BaseSaver:
    '''Base class to extact lines, labels from matplotlib ContourSet'''
    def __init__(self, contours: Contours, bestfit: Bestfit):
        self.x_points: NList = []
        self.y_points: NList = []
        self.labels: List[str] = []
        self.bestfit = bestfit
        self.extract_contours_levels(contours)

    def extract_contours_levels(self, contours: Contours):
        for line in contours.collections:
            try:
                path: MPath = line.get_paths()[0]
            except IndexError:
                continue
            vertices: np.ndarray = path.vertices
            x, y = vertices[:,0], vertices[:,1]
            self.x_points.append(x)
            self.y_points.append(x)
            self.labels.append(line.get_label())

class CustomDtypes:
    contour_dt: np.dtype = np.dtype([('x', np.float64), ('y', np.float64)])
    unc_dt: np.dtype  = np.dtype([('neg', np.float64), ('pos', np.float64)])
    bestfit_dt: np.dtype = np.dtype([('value', np.float64), ('unc', unc_dt)])

@savemap(("hdf5", "hdf"))
class HDFSaver(BaseSaver):
    '''Saver to HDF5'''
    def save_to_file(self, output: Path):
        out_file: h5py.File = h5py.File(output, "w")
        contours = out_file.create_group('contours')
        for xarr, yarr, label in zip(self.x_points, self.y_points, self.labels):
            contours[label] = np.core.records.fromarrays([xarr, yarr], dtype=CustomDtypes.contour_dt)
        out_file['/bestfit'] = np.array(self.bestfit, dtype=CustomDtypes.bestfit_dt)
        out_file.close()

@savemap("npz")
class NpzSaver(BaseSaver):
    '''Saver to npz format'''
    def save_to_file(self, output: Path):
        contours={}
        for xarr, yarr, label in zip(self.x_points, self.y_points, self.labels):
            contours[label] = np.core.records.fromarrays([xarr, yarr], dtype=CustomDtypes.contour_dt)
        contours['bestfit'] = np.array(self.bestfit, dtype=CustomDtypes.bestfit_dt)
        np.savez(output, **contours)
         
@savemap("root")
class ROOTSaver(BaseSaver):
    '''Saver to ROOT format'''
    def save_to_file(self, output: Path):
        out_file: ROOT.TFile = ROOT.TFile.Open(output.as_posix(), "recreate")
        for xarr, yarr, label in zip(self.x_points, self.y_points, self.labels):
            size: int = len(xarr)
            graph = ROOT.TGraph(size)
            for i, (x, y) in enumerate(zip(xarr, yarr)):
                graph.SetPoint(i, x, y)
                graph.SetName(label)
                graph.SetTitle(label)
            out_file.WriteTObject(graph)
        self._save_bestfit(out_file)
        out_file.Close()

    def _save_bestfit(self, tfile):
        bestfit_graph = ROOT.TGraphAsymmErrors(1)
        (x,xerr), (y,yerr) = self.bestfit
        bestfit_graph.SetPoint(0, x, y)
        bestfit_graph.SetTitle("bestfit")
        bestfit_graph.SetName("bestfit")
        if isinstance(xerr, tuple):
            bestfit_graph.SetPointEXlow(0, xerr[0])
            bestfit_graph.SetPointEXhigh(0, xerr[1])
        else:
            bestfit_graph.SetPointEXlow(0, xerr)
            bestfit_graph.SetPointEXhigh(0, xerr)
        if isinstance(yerr, tuple):
            bestfit_graph.SetPointEYlow(0, yerr[0])
            bestfit_graph.SetPointEYhigh(0, yerr[1])
        else:
            bestfit_graph.SetPointEYlow(0, yerr)
            bestfit_graph.SetPointEYhigh(0, yerr)
        tfile.WriteTObject(bestfit_graph)

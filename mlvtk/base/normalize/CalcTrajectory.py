import pathlib
from typing import List, Generator, Union
import h5py
import numpy as np
from ..Model import Model
from sklearn.decomposition import PCA


class CalcTrajectory:
    @staticmethod
    # TODO ADD ARGUMENTS
    def _build_item_list(
        directory: List[pathlib.Path],
    ) -> List[List[np.ndarray]]:
        # return list of list of numpy arrays.
        # representing list of model weights @ each epoch
        model_list: List[List[np.ndarray]] = []

        def _filter_datasets(
            name: str, obj: Union[h5py.Dataset, h5py.Group]
        ) -> Union[h5py.Dataset, None]:
            if isinstance(obj, h5py.Dataset):
                model_layers.append(obj[:])

        for f in directory:  # for file in directory
            model_layers: List[np.ndarray] = []
            h5obj = h5py.File(f, mode="r")
            h5obj.visititems(_filter_datasets)
            model_list.append(model_layers)
            h5obj.close()

        return model_list

    @staticmethod
    def sort_files(path: pathlib.Path) -> List[pathlib.Path]:
        return sorted(
            path.glob(r"model_[0-9]*"),
            key=lambda x: int(x.parts[-1].split("_")[-1][:-3]),
        )

    @staticmethod
    def _calc_weight_differences(
        modeldata: List[List[np.ndarray]],
    ) -> List[List[np.ndarray]]:
        theta_final: List[np.ndarray] = modeldata.pop()
        differences: List[List[np.ndarray]] = [
            [tf_w - tw for tf_w, tw in zip(theta_final, theta)] for theta in modeldata
        ]
        return differences

    @staticmethod
    def _get_T0(weight_diffs: List[List[np.ndarray]]) -> np.ndarray:
        flat_weight_diffs: np.ndarray = np.array(
            [
                np.concatenate([w.flatten() if w.ndim > 1 else w for w in weights])
                for weights in weight_diffs
            ]
        )
        return flat_weight_diffs

    @staticmethod
    def _get_path(dir_list) -> Generator[pathlib.Path, None, None]:
        for d in dir_list:
            assert d.is_dir(), f"{d} is not a directory!"
            yield d

    def _aggregate_files(self, obj):
        self.files = map(self.sort_files, self._get_path(obj))

    def _yield_file_lists(self) -> Generator[List[pathlib.Path], None, None]:
        for mod in self.files:
            yield mod  # TODO call _load_model() on mod

    def _get_raw_weights(self):  # for
        for modeldata in map(self._build_item_list, self._yield_file_lists()):
            yield modeldata

    def _get_weight_diffs(self):
        for weight_diffs in map(self._calc_weight_differences, self._get_raw_weights()):
            yield weight_diffs

    def get_T0(self)->np.ndarray:
        T0 = np.array(
            [
                flat_weight_diffs
                for flat_weight_diffs in map(self._get_T0, self._get_weight_diffs())
            ]
        )

        return T0

    def fit(self, obj: Union[List[Union[Model, pathlib.Path, str]], Model]):

        if isinstance(obj, List):
            if isinstance(obj[0], Model):
                _obj = [getattr(mod, "_get_cpoint_path")() for mod in obj]
                self._aggregate_files(_obj)
            elif isinstance(obj[0], pathlib.Path):
                self._aggregate_files(obj)
            else:
                _obj = [pathlib.Path(str(p)) for p in obj]
                self._aggregate_files(_obj)
        else:
            self._aggregate_files([obj._get_cpoint_path()])

        return self.get_T0()


import pathlib
from typing import Generator, List, Tuple, Union, TypeVar
import h5py
import numpy as np
from sklearn.decomposition import PCA

vmodel = TypeVar("vmodel")


class CalcTrajectory:
    def _build_item_list(
        self,
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

        self.dims = [(*i.shape, np.prod(i.shape)) for i in model_list[0]]

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
                np.concatenate([w.flatten() if w.ndim > 1 else w for w in weights])  # type: ignore
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
            yield mod  

    def _get_raw_weights(self):  # for
        for modeldata in map(self._build_item_list, self._yield_file_lists()):
            yield modeldata

    def _get_weight_diffs(self):
        for weight_diffs in map(self._calc_weight_differences, self._get_raw_weights()):
            yield weight_diffs

    def get_T0(self) -> np.ndarray:
        T0 = np.array(
            [
                flat_weight_diffs
                for flat_weight_diffs in map(self._get_T0, self._get_weight_diffs())
            ]
        )

        return T0

    def _component_allocation(self, component: np.ndarray) -> np.ndarray:
        l: List[np.ndarray] = []
        idx: int = 0
        for d in self.dims:
            l.append(component[idx : idx + d[-1]].flatten())
            idx += d[-1]

        return np.concatenate(l)  # type: ignore

    @staticmethod
    def _yield_model(t0) -> Generator[List[List[np.ndarray]], None, None]:
        for mod in t0:
            yield mod

    @staticmethod
    def project_2d(
        epoch_data: np.ndarray, xd: List[np.ndarray], yd: List[np.ndarray]
    ) -> Tuple[np.float32, np.float32]:  # type: ignore
        assert (
            len(epoch_data) == len(xd) == len(yd)
        ), f"dimensions do  not match\
                for epoch_data: {len(epoch_data)}, xd: {len(xd)}, yd: {len(yd)}"
        xd = np.array(xd)
        yd = np.array(yd)
        x = np.divide(np.dot(xd, epoch_data), np.linalg.norm(epoch_data))  # type: ignore
        y = np.divide(np.dot(yd, epoch_data), np.linalg.norm(epoch_data))  # type: ignore

        return x, y

    def fit(self, obj: Union[List[Union[vmodel, pathlib.Path, str]], vmodel]):

        if isinstance(obj, List):
            if isinstance(obj[0], str):
                _obj = [pathlib.Path(str(p)) for p in obj]
                self._aggregate_files(_obj)
            elif isinstance(obj[0], pathlib.Path):
                self._aggregate_files(obj)
            else:
                _obj = [mod._get_cpoint_path() for mod in obj]
                self._aggregate_files(_obj)
        else:
            self._aggregate_files([obj._get_cpoint_path()])

        T0: np.ndarray = self.get_T0()

        if not np.any(T0):
            self.xdir: np.ndarray = np.zeros((1))  # type: ignore
            self.ydir: np.ndarray = np.zeros((1))  # type: ignore
            print("WARNING::No weight change btw epochs")
            self.evr: List[np.nan] = [np.nan, np.nan]
            return
        xdir_list, ydir_list = [], []

        pca = PCA(n_components=2)
        if T0.ndim > 2:
            pca.fit(np.concatenate(T0))  # type: ignore
        else:
            pca.fit(T0)
        pca_1: np.ndarray = pca.components_[0]
        pca_2: np.ndarray = pca.components_[1]
        self.pca_dirs: List[np.ndarray] = pca.components_

        T0_models = map(self._yield_model, T0)

        for model_data in T0_models:
            model_xdir, model_ydir = [0], [0]
            for epoch_data in model_data:
                xdir, ydir = self.project_2d(
                    epoch_data,
                    self._component_allocation(pca_1),
                    self._component_allocation(pca_2),
                )
                model_xdir.append(xdir)
                model_ydir.append(ydir)
            xdir_list.append(model_xdir)
            ydir_list.append(model_ydir)
        if np.shape(xdir_list)[0] == 1:
            xdir_list = np.squeeze(xdir_list)
            ydir_list = np.squeeze(ydir_list)

        self.xdir = xdir_list
        self.ydir = ydir_list
        self.evr = pca.explained_variance_ratio_

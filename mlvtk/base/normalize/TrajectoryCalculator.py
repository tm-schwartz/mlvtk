import pathlib
import warnings
from typing import Generator, List, Tuple, TypeVar, Union

import h5py
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

vmodel = TypeVar("vmodel")


class TrajectoryCalculator:

    __slots__ = ("kernel_dims", "xdir", "ydir", "evr", "pca_dirs", "selected_model_fn")

    def _aggregate_files(self, folders):
        """Take folders list, gather and sort by epoch. Folders can be a list of paths for multiple models."""
        for sorted_folder in map(self._sort_files, self._get_path(folders)):
            yield sorted_folder

    @staticmethod
    def _sort_files(model_folder: pathlib.Path) -> List[pathlib.Path]:
        """Read contents of a model_folder and glob + sort the contents."""
        return sorted(
            model_folder.glob(r"model_[0-9]*"),
            key=lambda x: int(x.parts[-1].split("_")[-1][:-3]),
        )

    @staticmethod
    def _get_path(folders) -> Generator[pathlib.Path, None, None]:
        """Iterate over folders, verify a given folder is a dir."""
        for d in folders:
            assert d.is_dir(), f"{d} is not a directory!"
            yield d

    def _make_ext_link_file(self, sorted_model_files, index):
        """Create external_links for model data"""

        dataset_paths = []
        self.kernel_dims: List[Tuple[float]] = []

        ext_link_file = f"h5_ext_links_{index}.hdf5"

        def _get_paths(name, obj):
            if isinstance(obj, h5py.Dataset):
                dataset_paths.append(obj.name)
                self.kernel_dims.append(obj.shape)

        with h5py.File(
            sorted_model_files[0], "r", libver="latest"
        ) as epoch_0_h5:  # TODO check if need libver
            epoch_0_h5.visititems(_get_paths)
        size: int = sum(map(np.prod, self.kernel_dims))

        with h5py.File(
            ext_link_file, "w", libver="latest"
        ) as h5_ext:  # TODO check if need libver
            for epoch, epoch_file in enumerate(sorted_model_files):
                for dataset_num, ds_path in enumerate(dataset_paths):
                    h5_ext[f"{epoch}.{dataset_num}"] = h5py.ExternalLink(
                        epoch_file, ds_path
                    )

        return size, ext_link_file, index, sorted_model_files[-1], dataset_paths

    def create_t0(
        self, size, ext_link_file, index, final_model_fn, dataset_paths, selected_model
    ):
        def sub(ms):
            minuend, subtrahend = ms
            return np.subtract(minuend[:], subtrahend[:]).flatten()

        fin_model_fn_split = int(final_model_fn.stem.split("_")[-1])
        fin_model_fn_joined_epoch = final_model_fn.parent.joinpath(
            f"model_{selected_model}.h5"
        )

        if index == 0 and selected_model == fin_model_fn_split:
            self.selected_model_fn = fin_model_fn_joined_epoch

            selected_model_fn = final_model_fn.parent.joinpath(
                f"model_{selected_model}.h5"
            )
        elif index > 0 and selected_model == fin_model_fn_split:
            selected_model_fn = self.selected_model_fn
        else:
            selected_model_fn = final_model_fn.parent.joinpath(
                f"model_{selected_model}.h5"
            )

        # n_epochs: int = fin_model_fn_split
        with h5py.File(ext_link_file, "r", libver="latest") as elf:
            elf_keys = elf.keys()
            n_epochs = max([int(x.split(".")[0]) for x in elf_keys])
            with h5py.File(
                "t0_h5.hdf5", "a", libver="latest"
            ) as t0_h5:  # TODO check what happens when file already exists
                with h5py.File(selected_model_fn, "r", libver="latest") as sel_model:
                    t0_ds = t0_h5.create_dataset(
                        f"t0_{index}",
                        shape=(1, size),
                        maxshape=(n_epochs + 1, size),
                        fillvalue=0.0,
                        dtype="float32",
                    )
                    for link in tqdm(iterable=range(n_epochs + 1), leave=False):
                        data_list = [
                            (
                                elf[f"{link}.{i}"],
                                sel_model[dsp],
                            )
                            for i, dsp in enumerate(dataset_paths)
                        ]
                        if link > 0 and link <= n_epochs:
                            t0_ds.resize((link + 1, size))
                        t0_ds[link, 0:size] = np.concatenate([*map(sub, data_list)])
                        if not t0_ds.attrs.keys() and (
                            np.any(np.isnan(t0_ds[link][:]))
                            or np.any(np.isnan(t0_ds[link][:]))
                        ):
                            t0_ds.attrs.create("isnull", 1)

    @staticmethod
    def _slice_generator(dataset):
        """Very naive 'chunking' for incrementalPCA. Just take two rows, and if """
        offset = 2
        for row in range(0, len(dataset), 2):
            if row == len(dataset) - 3:
                return dataset[row:]
            yield dataset[row : row + offset]

    @staticmethod
    def _yield_util(fun, inpt):
        for indice, data in enumerate(inpt):
            yield fun(data, indice)

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
        x = np.divide(np.dot(epoch_data, xd), np.linalg.norm(xd))  # type: ignore
        y = np.divide(np.dot(epoch_data, yd), np.linalg.norm(yd))  # type: ignore

        return x, y

    def fit(
        self,
        obj: Union[List[Union[vmodel, pathlib.Path, str]], vmodel],
        selected_model=0,
    ):

        # check for `t0_h5.hdf5`
        t0_h5 = pathlib.Path("t0_h5.hdf5")
        if t0_h5.exists():
            pathlib.os.remove(t0_h5)

        if isinstance(obj, List):
            if isinstance(obj[0], str):
                _obj = [pathlib.Path(str(p)) for p in obj]
                data = self._aggregate_files(_obj)
            elif isinstance(obj[0], pathlib.Path):
                data = self._aggregate_files(obj)
            else:
                _obj = [mod._get_cpoint_path() for mod in obj]
                data = self._aggregate_files(_obj)
        else:
            data = self._aggregate_files([obj._get_cpoint_path()])

        for path_data in self._yield_util(self._make_ext_link_file, data):
            self.create_t0(*path_data, selected_model=selected_model)
        self.xdir = []
        self.ydir = []
        self.evr = []
        self.pca_dirs = []
        xdir_list, ydir_list = [], []
        ipca = IncrementalPCA(2)
        with h5py.File("t0_h5.hdf5", "r", libver="latest") as T0:
            for ds in tqdm(iterable=T0.keys(), leave=True):
                if T0[ds].attrs.keys():
                    self.xdir.append(np.zeros((1)))
                    self.ydir.append(np.zeros((1)))
                    self.evr.append([np.nan, np.nan])
                    print(f"WARNING::No weight change btw epochs or nan for {ds}")
                else:
                    for data in tqdm(
                        self._slice_generator(T0[ds]),
                        leave=False,
                    ):
                        ipca.partial_fit(data)

            pca_1: np.ndarray = ipca.components_[0]
            pca_2: np.ndarray = ipca.components_[1]
            self.pca_dirs: List[np.ndarray] = ipca.components_

            for model in T0.keys():
                model_xdir, model_ydir = [], []
                # final_epoch = T0[model][-1]
                # for epoch in range(len(T0[model])- 1):
                for epoch in T0[model]:
                    #    epoch_diff = final_epoch - T0[model][epoch]
                    # manually calculate pca transform in case of data being too large for ram
                    xdir, ydir = self.project_2d(
                        epoch,
                        pca_1,
                        pca_2,
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
            self.evr = ipca.explained_variance_ratio_

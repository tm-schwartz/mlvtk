from typing import List, TypeVar, Union
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from tensorflow import data as tfdata
from tensorflow import device

T = TypeVar("T")
LLNP = Union[List[np.float32], List[List[np.float32]]]  # type: ignore


def _linspace_dispatcher(j, jdiff, jsize):
    return np.linspace(j.min() - jdiff, j.max() + jdiff, num=jsize, dtype=np.float32)  # type: ignore


def _normalize_filter(fw: np.ndarray) -> np.ndarray:
    f, w = fw
    return f * np.linalg.norm(w) / (np.linalg.norm(f) + 1e-10)


def _evaluate(model, filters, data):
    model.set_weights(filters)
    if not isinstance(data, tfdata.Dataset):
        return model.evaluate(
            x=data[0], y=data[1], use_multiprocessing=True, verbose=0, return_dict=True
        ).get("loss")

    return model.evaluate(
        data, use_multiprocessing=True, verbose=0, return_dict=True
    ).get("loss")


def normalizer(
    model,
    traj_calc=None,
    alphas_size: int = 35,
    betas_size: int = 35,
    extension: T = 1,
    quiet=False,
    **kwargs,
):

    weights: List[np.ndarray] = model.get_weights()

    for key in kwargs:
        assert key in [
            "alphas",
            "betas",
            "xdir",
            "ydir",
            "pca_dirs",
            "alphas_size",
            "betas_size"
        ], f"{key} is not a\
        valid kwarg"

    assert (
        traj_calc is not None or kwargs.get("xdir") is not None
    ), "Need traj_calc or xdir/ydir"

    if traj_calc:
        xdir, ydir = traj_calc.xdir, traj_calc.ydir
        pca_dirs = traj_calc.pca_dirs

    else:
        xdir, ydir = kwargs["xdir"], kwargs["ydir"]
        if "pca_dirs" in kwargs:
            pca_dirs = kwargs["pca_dirs"]
        else:
            pca_dirs = None

    if np.size(xdir) == np.size(ydir) and np.size(xdir) <= 1:  # type: ignore
        alphas = np.linspace(-5, 5, num=alphas_size, dtype=np.float32)  # type: ignore
        betas = np.linspace(-5, 5, num=betas_size, dtype=np.float32)  # type: ignore

    elif extension == "std":
        xdiff = np.std(xdir)
        ydiff = np.std(ydir)

    elif extension == 1:
        xdiff = ydiff = 1

    else:
        xdiff, ydiff = extension

    if np.ndim(xdir) > 1:
        concat_xdir = np.concatenate(xdir)  # type: ignore
        concat_ydir = np.concatenate(ydir)  # type: ignore
        alphas = _linspace_dispatcher(concat_xdir, xdiff, alphas_size)  # type: ignore
        betas = _linspace_dispatcher(concat_ydir, ydiff, betas_size)  # type: ignore
        coordinate_list = [np.column_stack([a, b]) for a, b in zip(xdir, ydir)]

    else:
        alphas = _linspace_dispatcher(xdir, xdiff, alphas_size)  # type: ignore
        betas = _linspace_dispatcher(ydir, ydiff, betas_size)  # type: ignore
        coordinate_list = np.column_stack([xdir, ydir])

    # For readability, make local vars
    alph = alphas[None, :, None]  # [[[`val`], [`val`],...]]
    bet = betas[:, None, None]  # [[[`val`]], [[`val`]]...]
    bet_zeros = np.zeros_like(bet)
    alph_zeros = np.zeros_like(alph)

    # in order to construct cartesian product:
    # take advantage of broadcasting with tf.add.
    # example:::
    # alphas.shape -> (50,)
    # betas.shape -> (50,)
    # alph.shape/alph_zeros.shape -> (1, 50, 1)
    # bet.shape/bet_zeros.shape -> (50, 1, 1)
    #
    # By doing sum_alph = tf.add(alph, bet_zeros) and sum_bet = tf.add(bet, alph_zeros)
    # we get two tensors of shape (50, 50, 1).
    # Then we tf.concat([sum_alph, sum_bet], axis=2).
    # By doing it along axis=2, we get pairs of values
    # in shape (50, 50, 2). Finally, we use tf.reshape
    # to get final cartesian product in result.shape = (2500, 2),
    # so then result[i] == [`val_i_0`, `val_i_1`]
    # adapted from https://stackoverflow.com/a/50195230
    alphas_betas_list = np.reshape(
        np.concatenate([alph + bet_zeros, alph_zeros + bet], axis=2),  # type: ignore
        (alphas.shape[0] * betas.shape[0], 2),
    )
    # alphas_betas_list = np.column_stack([alphas, betas])

    if pca_dirs is None:
        # gamma <- delta
        # delta <- eta
        gamma = np.array([np.reshape(np.random.randn(ww.size), ww.shape) for ww in weights], dtype="object")  # type: ignore
        delta = np.array([np.reshape(np.random.randn(ww.size), ww.shape) for ww in weights], dtype="object")  # type: ignore
    else:
        pca_gamma, pca_delta = pca_dirs
        gamma = []
        delta = []
        i = 0
        for l in weights:
            gamma.append(_normalize_filter((pca_gamma[i : i + l.size], l)))
            delta.append(_normalize_filter((pca_delta[i : i + l.size], l)))
            i += l.size

    batch_norm_filter = filter(
        lambda layer: "batch_normalization" in layer.name, model.layers
    )
    if batch_norm_filter:
        for layer in batch_norm_filter:
            i = model.layers.index(layer)
            gamma[i] = np.zeros((gamma[i].shape))  # type: ignore
            delta[i] = np.zeros((delta[i].shape))  # type: ignore

    w_concat = np.concatenate([w.flatten() for w in weights])
    g_concat = np.concatenate(gamma)
    d_concat = np.concatenate(delta)

    def _reshape(values):
        s = 0
        nw = []
        for w in weights:
            nw.append(np.reshape(values[s : s + w.size], w.shape))
            s += w.size
        return nw

    def _yield(arr):
        for (a, b) in arr:
            yield _reshape(w_concat + (g_concat * a + d_concat * b))

    normalized_filters = _yield(alphas_betas_list)

    non_eager_model = model._new_model()

    if np.ndim(coordinate_list) > 2:
        optimizer_filters = [_yield(coord_list) for coord_list in coordinate_list]
        optimizer_losses = []

        for mod in tqdm(optimizer_filters, desc="optimizer path", disable=quiet):
            t = []
            for filt in tqdm(mod, desc="filter", disable=quiet):
                t.append(
                    _evaluate(non_eager_model, filt, weights, model.validation_data)
                )
            optimizer_losses.append(t)

    else:
        optimizer_filters = _yield(coordinate_list)
        optimizer_losses = np.zeros((np.size(xdir)))  # type: ignore

        for i, filt in enumerate(
            tqdm(optimizer_filters, desc="filter", disable=quiet, total=np.size(xdir))
        ):
            optimizer_losses[i] = _evaluate(
                non_eager_model, filt, model.validation_data
            )

    total = alphas_size * betas_size
    filter_losses = np.zeros((total))

    for i, filt in enumerate(
        tqdm(normalized_filters, desc=f"filter", disable=quiet, total=total)
    ):
        filter_losses[i] = _evaluate(non_eager_model, filt, model.validation_data)

    return {
        "surface": pd.DataFrame(
            data=np.reshape(filter_losses, (np.size(alphas), np.size(betas))),
            index=alphas,
            columns=betas,
        ),
        "optimizer": (xdir, ydir, optimizer_losses),
    }

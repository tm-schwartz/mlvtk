import functools
from typing import List, TypeVar, Union, Optional
from tqdm import tqdm
import numpy as np

T = TypeVar("T")
LLNP = Union[List[np.float32], List[List[np.float32]]]  # type: ignore


def _linspace_dispatcher(j, jdiff, jsize):
    return np.linspace(j.min() - jdiff, j.max() + jdiff, num=jsize, dtype=np.float32)  # type: ignore


def _reshape(values, weights):
    s = 0
    for w in weights:
        yield np.reshape(values[s : s + w.size], w.shape)
        s += w.size


def _normalize_filter(fw: np.ndarray) -> np.ndarray:
    # TODO: VERIFY NEED FOR ARRAY CAST
    f, w = fw
    return f * np.array([(np.linalg.norm(w) / (np.linalg.norm(f) + 1e-10))])


def _gen_filters(arr1, arr2, arr3, weights):
    return np.array(
        [
            np.array(
                [
                    h * a + i * b
                    for h, i in zip(
                        map(_normalize_filter, zip(arr1, weights)),
                        map(_normalize_filter, zip(arr2, weights)),
                    )
                ],
                dtype="object",
            )
            for a, b in arr3
        ],
        dtype="object",
    )


def _yield(arr):
    for a in arr:
        yield a

def _evaluate(model, filters, weights):
    new_weights = np.add(weights, filters) # type: ignore
    model.set_weights(new_weights)
    return model.evaluate(*model.validation_data,
                use_multiprocessing=True,
                verbose=0)

def normalizer(
    model,
    traj_calc=None,
    alphas_size: int = 35,
    betas_size: int = 35,
    extension: T = 1,
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

    if len(xdir) == len(ydir) <= 1:  # type: ignore
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
        np.concatenate([alph + bet_zeros, alph_zeros + bet], axis=2), # type: ignore
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
        gamma = np.array([pg for pg in _reshape(pca_gamma, weights)])
        delta = np.array([pd for pd in _reshape(pca_delta, weights)])

    batch_norm_filter = filter(
        lambda layer: "batch_normalization" in layer.name, model.layers
    )
    if batch_norm_filter:
        for layer in batch_norm_filter:
            i = model.layers.index(layer)
            gamma[i] = np.zeros((gamma[i].shape))  # type: ignore
            delta[i] = np.zeros((delta[i].shape))  # type: ignore

    normalized_filters = _gen_filters(gamma, delta, alphas_betas_list, weights).reshape(
        alphas.shape[0], betas.shape[0], len(weights)
    )

    if np.ndim(coordinate_list) > 2:
        optimizer_filters = [
                _gen_filters(gamma, delta, coord_list, weights)
                for coord_list in coordinate_list
            ]
        optimizer_losses = []

        for mod in tqdm(optimizer_filters):
            for filt in tqdm(mod):
                optimizer_losses.append(_evaluate(model, filt, weights,))

    else:
        optimizer_filters = _gen_filters(gamma, delta, coordinate_list, weights)
        optimizer_losses = np.zeros((np.size(xdir))) # type: ignore

        for i, filt in enumerate(tqdm(optimizer_filters)):
            optimizer_losses[i] = _evaluate(model, filt, weights)

    

    filter_losses =[]
    for ai in tqdm(_yield(normalized_filters)):
        for filt in tqdm(ai):
            filter_losses.append(_evaluate(model, filt, weights))

    return np.reshape(filter_losses, (np.size(alphas), np.size(betas)))





            



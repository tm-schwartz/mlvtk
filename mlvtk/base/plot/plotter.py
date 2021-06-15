import plotly.graph_objects as go
from pandas import DataFrame
import numpy as np
from scipy import interpolate


def make_trace(
    data,
    opacity=0.9,
    coloraxis="coloraxis",
    lighting=dict(ambient=0.6, roughness=0.9, diffuse=0.5, fresnel=2),
    surf_name="loss surface",
    marker=dict(symbol="circle", size=3),
    line=dict(color="firebrick", width=2),
    showlegend=True,
    scatter_name="path",
    resolution=200,
    scaling=lambda z: np.log(z) / np.log(1.5),
):
    if isinstance(data, DataFrame):
        xc, yc = np.meshgrid(data.index, data.columns)
        z_vals = data.values
        spline = interpolate.RectBivariateSpline(xc[0, :], yc[:, 0], z_vals, s=0)
        x_arr = xc.ravel()
        y_arr = yc.ravel()
        z_arr = z_vals.ravel()
        x_linspc = np.linspace(min(x_arr), max(x_arr), resolution)
        y_linspc = np.linspace(min(y_arr), max(y_arr), resolution)
        z_spline = spline(x_linspc, y_linspc).ravel()
        x_coords, y_coords = np.meshgrid(x_linspc, y_linspc)
        final_x_coords = x_coords.ravel()
        final_y_coords = y_coords.ravel()
        z_spline[z_spline < 0] = 0.0  # change neg to 0. avoid runtime warn
        z_values = [scaling(z) for z in z_spline]

        mesh_trace = go.Mesh3d(
            x=final_x_coords,
            y=final_y_coords,
            z=z_values,
            intensity=z_values,
            opacity=opacity,
            coloraxis=coloraxis,
            lighting=lighting,
            name=surf_name,
        )
        return mesh_trace
    elif isinstance(data, tuple):
        if np.ndim(data[0]) > 1:
            colors = [
                f"rgb{(np.random.randint(256), np.random.randint(256), np.random.randint(256))}"
                for _ in range(len(data[0]))
            ]

            scatter_trace = tuple(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=[scaling(z_val) for z_val in z],
                    marker=dict(**marker, color=color),
                    line=line,
                    showlegend=showlegend,
                    name=f"{scatter_name}_{i}",
                )
                for i, (x, y, z, color) in enumerate(zip(*data, colors))
            )

        else:
            scatter_trace = go.Scatter3d(
                x=data[0],
                y=data[1],
                z=[scaling(z_val) for z_val in data[2]],
                marker=marker,
                line=line,
                showlegend=showlegend,
                name=scatter_name,
            )
        return scatter_trace


def make_figure(traces):

    banned = [
        "rgb(41, 24, 107)",
        "rgb(42, 35, 160)",
        "rgb(15, 71, 153)",
        "rgb(18, 95, 142)",
        "rgb(38, 116, 137)",
        "rgb(53, 136, 136)",
        "rgb(65, 157, 133)",
        "rgb(81, 178, 124)",
        "rgb(111, 198, 107)",
        "rgb(160, 214, 91)",
        "rgb(212, 225, 112)",
        "rgb(253, 238, 153)",
    ]

    if isinstance(traces[1], tuple):
        traces = [traces[0], *traces[1]]

    for trace in traces[1:]:
        if trace.marker.color in banned:
            trace.update(
                marker=f"rgb{(np.random.randint(256), np.random.randint(256), np.random.randint(256))}"
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        autosize=False,
        width=1200,
        height=900,
        # margin=dict(l=10),
        legend=dict(yanchor="top", y=0.89, xanchor="right", x=0.8),
        # bargap=0.2,
        coloraxis=dict(
            colorscale="haline_r",
            colorbar=dict(
                title="Loss Surface",
                len=0.45,
                yanchor="top",
                y=0.79,
                xanchor="right",
                x=0.805,
            ),
        ),
    )
    return fig

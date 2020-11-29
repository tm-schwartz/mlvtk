import plotly.graph_objects as go
from pandas import DataFrame
import numpy as np
import pprint


def make_trace(
    data,
    opacity=0.9,
    coloraxis="coloraxis",
    lighting=dict(ambient=0.6, roughness=0.9, diffuse=0.5, fresnel=2),
    surf_name="loss surface",
    marker=dict(symbol="circle", size=3),
    line=dict(color="darkblue", width=2),
    showlegend=True,
    scatter_name="path",
):
    if isinstance(data, DataFrame):
        surface_trace = go.Surface(
            x=data.index,
            y=data.columns,
            z=data.values,
            opacity=opacity,
            coloraxis=coloraxis,
            lighting=lighting,
            name=surf_name,
        )
        return surface_trace
    elif isinstance(data[0], tuple):
        if len(data[0]) > 1:
            colors = [
                f"rgb{(np.random.randint(256), np.random.randint(256), np.random.randint(256))}"
                for _ in range(len(data[0]))
            ]

            breakpoint()
            scatter_trace = tuple(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    marker=dict(**marker, color=color),
                    line=line,
                    showlegend=showlegend,
                    name=f"{scatter_name}_{i}",
                )
                for i, (x, y, z, color) in enumerate(
                    zip(*data, colors)
                )
            )

    else:
            scatter_trace = go.Scatter3d(
                x=data[0],
                y=data[1],
                z=data[2],
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
        margin=dict(l=10),
        bargap=0.2,
        coloraxis=dict(
            colorscale="haline_r",
            colorbar=dict(title="Loss Surface Value", len=0.45),
        ),
            legend={"itemsizing": "constant"},
    )
    return fig

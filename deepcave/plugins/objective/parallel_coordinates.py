#  noqa: D400

"""
# ParallelCoordinates

This module provides utilities for visualizing the parallel coordinates.
The labels are of the important hyperparameters (HPs), budget and objectives.

## Classes
    - ParallelCoordinates : Can be used for visualizing the parallel coordinates.
"""

from typing import Any, Callable, Dict, List, Union

from collections import defaultdict

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from dash.exceptions import PreventUpdate

from deepcave import config
from deepcave.constants import VALUE_RANGE
from deepcave.evaluators.fanova import fANOVA
from deepcave.plugins.static import StaticPlugin
from deepcave.utils.compression import deserialize, serialize
from deepcave.utils.layout import get_checklist_options, get_select_options, help_button
from deepcave.utils.logs import get_logger
from deepcave.utils.styled_plotty import get_hyperparameter_ticks, save_image

logger = get_logger(__name__)


class ParallelCoordinates(StaticPlugin):
    """
    Can be used for visualizing the parallel coordinates.

    Attributes
    ----------
    id : str
        The identificator of the plugin.
    name : str
        The name of the plugin.
    icon : str
        The icon representing the plugin
    activate_run_selection : bool
        Define whether the run selection feature is active.
    help : str
        The path to the documentation of the plugin.
    """

    id = "parallel_coordinates"
    name = "Parallel Coordinates"
    icon = "far fa-map"
    activate_run_selection = True
    help = "docs/plugins/parallel_coordinates.rst"

    @staticmethod
    def get_input_layout(register: Callable) -> List[Union[html.Div, dbc.Row]]:
        """
        Get the input layout as html container and dash bootstrap component (DBC).

        Parameters
        ----------
        register : Callable
            Used to get the id for the select object.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Union[html.Div, dbc.Row]]
            An html container and a dash bootstrap component (DBC) of the layout of the input.
        """
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Objective"),
                            dbc.Select(
                                id=register("objective_id", ["value", "options"], type=int),
                                placeholder="Select objective ...",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Budget"),
                            help_button(
                                "Combined budget means that the trial on the highest evaluated"
                                " budget is used.\n\n"
                                "Note: Selecting combined budget might be misleading if a time"
                                " objective is used. Often, higher budget take longer to evaluate,"
                                " which might negatively influence the results."
                            ),
                            dbc.Select(
                                id=register("budget_id", ["value", "options"], type=int),
                                placeholder="Select budget ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dbc.Label("Show Important Hyperparameters"),
                    help_button(
                        "Only the most important hyperparameters are shown which are "
                        "calculated by fANOVA using 10 trees. The more left a "
                        "hyperparameter stands, the more important it is. However, activating "
                        "this option might take longer."
                    ),
                    dbc.Select(
                        id=register("show_important_only", ["value", "options"]),
                        placeholder="Select ...",
                    ),
                ]
            ),
        ]

    @staticmethod
    def get_filter_layout(register: Callable) -> List[Union[dbc.Row, html.Div]]:
        """
        Get the filtered layout for a dash bootstrap component (DBC) and html container.

        Parameters
        ----------
        register : Callable
            Used for the id of the Input, select object and html Checklist.
            The register_input function is located in the Plugin superclass.

        Returns
        -------
        List[Union[dbc.Row, html.Div]]
            A filtered dash bootstrap component (DBC) and html container.
        """
        return [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Limit Hyperparameters"),
                            help_button(
                                "Shows either the n most important hyperparameters (if show "
                                "importance hyperparameters is true) or the first n selected "
                                "hyperparameters."
                            ),
                            dbc.Input(id=register("n_hps", "value"), type="number"),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Show Unsuccessful Configurations"),
                            help_button("Whether to show all configurations or only failed ones."),
                            dbc.Select(
                                id=register("show_unsuccessful", ["value", "options"]),
                                placeholder="Select ...",
                            ),
                        ],
                        md=6,
                    ),
                ],
            ),
            html.Div(
                [
                    dbc.Label("Hyperparameters"),
                    dbc.Checklist(
                        id=register("hyperparameter_names", ["value", "options"]), inline=True
                    ),
                ],
                className="mt-3",
                id=register("hide_hps", ["hidden"]),
            ),
        ]

    def load_inputs(self) -> Dict[str, Dict[str, Any]]:
        """Load the inputs containing information about the values to be visualized."""
        return {
            "show_important_only": {"options": get_select_options(binary=True), "value": "true"},
            "show_unsuccessful": {"options": get_select_options(binary=True), "value": "false"},
            "n_hps": {"value": 0},
            "hyperparameter_names": {"options": get_checklist_options(), "value": []},
            "hide_hps": {"hidden": True},
        }

    # Types dont match superclass
    def load_dependency_inputs(self, run, _, inputs):
        """
        Load the objective, budgets and hyperparameters (HPs) and its attributes.

        Parameters
        ----------
        run
            The run to get the objective, budget and hyperparameters (HPs) from.
        inputs
            Contains information about the objective, budget, configurations
            and hyperparameters (HPs).

        Returns
        -------
        The objective, budgets, hyperparameters (HPs) and their attributes.
        """
        # Prepare objectives
        objective_names = run.get_objective_names()
        objective_ids = run.get_objective_ids()
        objective_options = get_select_options(objective_names, objective_ids)
        objective_value = inputs["objective_id"]["value"]

        # Prepare budgets
        budgets = run.get_budgets(human=True)
        budget_ids = run.get_budget_ids()
        budget_options = get_checklist_options(budgets, budget_ids)
        budget_value = inputs["budget_id"]["value"]

        # Prepare others
        n_hps = inputs["n_hps"]["value"]
        hp_names = run.configspace.get_hyperparameter_names()

        if inputs["show_important_only"]["value"] == "true":
            hp_options = []
            hp_value = inputs["hyperparameter_names"]["value"]
            hidden = True
        else:
            hp_options = get_select_options(hp_names)

            values = inputs["hyperparameter_names"]["value"]
            if len(values) == 0:
                values = hp_names

            hp_value = values
            hidden = False

        if objective_value is None:
            objective_value = objective_ids[0]
            budget_value = budget_ids[-1]
            hp_value = hp_names
            n_hps = len(hp_names)

        if n_hps == 0:
            n_hps = len(hp_names)

        return {
            "objective_id": {
                "options": objective_options,
                "value": objective_value,
            },
            "budget_id": {
                "options": budget_options,
                "value": budget_value,
            },
            "hyperparameter_names": {
                "options": hp_options,
                "value": hp_value,
            },
            "n_hps": {"value": n_hps},
            "hide_hps": {"hidden": hidden},
        }

    @staticmethod
    # Types dont match superclass
    def process(run, inputs):
        """
        Get a serialized dataframe of the data and run a fanova for evaluation.

        Parameters
        ----------
        run : AbstractRun
            The run to process.
        inputs : Dict[str, Any]
            The inputs for the visualization.

        Returns
        -------
        result
            The serialized dataframe.
        """
        budget = run.get_budget(inputs["budget_id"])
        objective = run.get_objective(inputs["objective_id"])
        df = serialize(run.get_encoded_data(objective, budget))
        result: Dict[str, Any] = {"df": df}

        if inputs["show_important_only"]:
            # Let's run a quick fANOVA here
            evaluator = fANOVA(run)
            evaluator.calculate(objective, budget, n_trees=10, seed=0)
            importances_dict = evaluator.get_importances()
            importances = {u: v[0] for u, v in importances_dict.items()}
            important_hp_names = sorted(importances, key=lambda key: importances[key], reverse=True)
            result["important_hp_names"] = important_hp_names

        return result

    @staticmethod
    def get_output_layout(register: Callable) -> dcc.Graph:
        """
        Get the dash Graph output layout.

        Parameters
        ----------
        register : Callable
            Used for the id of the output object.
            The register_output function is located in the Plugin superclass.

        Returns
        -------
        dcc.Graph
            The dash Graph output layout.
        """
        return dcc.Graph(register("graph", "figure"), style={"height": config.FIGURE_HEIGHT})

    @staticmethod
    # Types dont match superclass
    def load_outputs(run, inputs, outputs):
        """
        Load and save the output figure.

        Parameters
        ----------
        run
            The run to be analyzed.
        inputs
            The inputs containing information about what to visualize.
        outputs
            Contains the serialized dataframe as well as the important
            hyperparameter (HPs) names.

        Returns
        -------
        The output figure.
        """
        objective = run.get_objective(inputs["objective_id"])
        objective_name = objective.name

        show_important_only = inputs["show_important_only"]
        show_unsuccessful = inputs["show_unsuccessful"]
        n_hps = inputs["n_hps"]

        if n_hps == "" or n_hps is None:
            raise PreventUpdate
        else:
            n_hps = int(n_hps)

        if show_important_only:
            hp_names = outputs["important_hp_names"]
        else:
            hp_names = inputs["hyperparameter_names"]

        hp_names = hp_names[:n_hps]

        df = outputs["df"]
        df = deserialize(df, dtype=pd.DataFrame)
        objective_values = []
        for value in df[objective_name].values:
            b = np.isnan(value)
            if not show_unsuccessful:
                b = not b
            if b:
                objective_values += [value]

        data: defaultdict = defaultdict(dict)
        for hp_name in hp_names:
            values = []
            for hp_v, objective_v in zip(df[hp_name].values, df[objective_name].values):
                b = np.isnan(objective_v)
                if not show_unsuccessful:
                    b = not b
                if b:
                    values += [hp_v]

            data[hp_name]["values"] = values
            data[hp_name]["label"] = hp_name
            data[hp_name]["range"] = VALUE_RANGE

            hp = run.configspace.get_hyperparameter(hp_name)
            tickvals, ticktext = get_hyperparameter_ticks(hp, ticks=4, include_nan=True)

            data[hp_name]["tickvals"] = tickvals
            data[hp_name]["ticktext"] = ticktext

        if show_unsuccessful:
            line = dict()
        else:
            data[objective_name]["values"] = objective_values
            data[objective_name]["label"] = objective_name
            line = dict(
                color=data[objective_name]["values"],
                showscale=True,
                colorscale="aggrnyl",
            )

        figure = go.Figure(
            data=go.Parcoords(
                line=line,
                dimensions=list([d for d in data.values()]),
                labelangle=45,
            ),
            layout=dict(margin=dict(t=150, b=50, l=100, r=0)),
        )
        save_image(figure, "parallel_coordinates.pdf")

        return figure

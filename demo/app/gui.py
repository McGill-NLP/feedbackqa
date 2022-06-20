import json
import time
from typing import List

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, State, ALL, MATCH
from dash_extensions.enrich import Trigger, Output
from jupyter_dash import JupyterDash
from pyngrok import ngrok
import random
from .utils import convert_to_md

from dash.long_callback import DiskcacheLongCallbackManager

## Diskcache
import diskcache

def answer_component(
    n,
    source=None,
    retrieved_answer="Answer will appear here...",
    rating=None,
    generated_feedback="Feedback will appear here...",
):
    answer_card = dbc.Card(
        [
            dbc.CardHeader(f"Retrieved Answer #{n}"),
            dbc.Spinner(
                dbc.CardBody(
                    [
                        html.H4(
                            source,
                            id={"name": "title-answer", "index": n},
                            className="card-title",
                        ),
                        dbc.Collapse(
                            dcc.Markdown(
                                retrieved_answer,
                                id={"name": "markdown-preview", "index": n},
                            ),
                            id={"name": "collapse-preview", "index": n},
                            is_open=True,
                        ),
                        dbc.Collapse(
                            dcc.Markdown(
                                retrieved_answer,
                                id={"name": "markdown-answer", "index": n},
                            ),
                            id={"name": "collapse-answer", "index": n},
                            is_open=False,
                        ),
                        dbc.Button(
                            "Show More",
                            id={"name": "show-more-btn", "index": n},
                            className="mb-3",
                            color="primary",
                            outline=True
                        ),
                    ],
                    className="card-text",
                )
            ),
        ],
        color="primary",
        outline=True,
        style={"height": "100%"},
    )

    feedback_card = dbc.Card(
        [
            dbc.CardHeader(f"Generated Feedback #{n}"),
            dbc.Spinner(
                dbc.CardBody(
                    [
                        html.H4(
                            rating,
                            id={"name": "title-feedback", "index": n},
                            className="card-title",
                        ),
                        dcc.Markdown(
                            generated_feedback,
                            id={"name": "markdown-feedback", "index": n},
                        ),
                    ],
                    className="card-text",
                )
            ),
        ],
        color="info",
        outline=True,
        style={"height": "100%"},
    )

    return dbc.Row(
        [
            dbc.Col(answer_card, width=7),
            dbc.Col(feedback_card, width=5),
        ],
        style={"padding-top": "15px", "padding-bottom": "15px"},
    )


def layout(sample_questions: List[str]) -> dbc.Container:
    navbar = dbc.Navbar(
        dbc.Container([
            #html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(dbc.NavbarBrand(
                            "FeedbackQA System", className="brand",
                            style={"color": "#008b8b", 
                                    "font-family": "Verdana, Helvetica, sans-serif", 
                                    "font-size": "22px", "font-weight": "bold",
                                    "padding-left": "2"}), 
                        align="start"),

                    ],
                    align="center",
                    justify="between",
                    className="g-0",
                ),
                #dbc.Col(
                html.H6(
                    dbc.Badge(
                            "McGill-NLP", color="danger", 
                            href='https://github.com/McGill-NLP/'),
                    style={"padding-top": "15px"}
                    ),
                    #, align="end"
                    #),
                #style={"textDecoration": "none"},
            #)
            
        ]),
        #color="primary",
        style={'color': "#cccccc"},
        dark=True,
        #brand="❓ FeedbackQA System",
        #brand=html.Div([html.H3(["❓ FeedbackQA System", dbc.Badge("by McGill-NLP", className="ms-badge")])]),
    )

    intro = html.Div(
        [  
            html.Div([
                html.Aside(
                    style={
                            'width': '25%',
                            'padding-left': '.5rem',
                            'margin-left': '.5rem',
                            'float': 'right',
                            'box-shadow': 'inset 5px 0 5px -5px #29627e',
                            'font-style': 'italic',
                            'color': '#29627e',
                        },
                    children=[
                            html.P("NOTE: Although the collected dataset contains COVID-related information, "
                                "it is only for research purposes and is not intended to provide medical advice.")
                                ]
                        ),
                html.P(
                        "FeedbackQA is a research prototype to show that interactive feedback is useful to improve QA models post-deployment. "
                        "We train our models on already collected interactive feedback data. " 
                        "Try asking a question about COVID-19, our system will retrieve passages as answers, and generate human-like feedback about the correctness of each answer, which is also used to rerank the answers."
                        )
                #html.Article(
                #    style={
                #            'margin': '0',
                #            'padding': '.3rem',
                #            'background-color': '#eee',
                #            'font': "1rem 'Fira Sans', sans-serif",
                #        },
                #    children=[
                #        html.P(
                #            "FeedbackQA is a research prototype that makes use of interactive feedback to improve QA models post-deployment."
                #            "Type in a question about COVID-19, our system selects the best passages as answer, and also genereates humna-like feedback"
                #            ", which are used for rerank the answers.")
                    ],
                className="lead"),
            #)]),
            #html.Div(
            dbc.Button("Paper", href='https://arxiv.org/abs/2204.03025', color="info", outline=True),
            html.A(" "),
            dbc.Button("Project page", href='https://mcgill-nlp.github.io/feedbackqa/', outline=True, color="info"),
            html.A(" "),
            dbc.Button("Code & Data", href='https://github.com/McGill-NLP/feedbackqa', outline=True, color="info"),
            html.A(" "),
            #html.Div([
            dbc.Button("More info", id="modal_open", color="info"),
            dbc.Modal(
                [
                    dbc.ModalHeader("Q&A about FeedbackQA"),
                    dbc.ModalBody([
                        html.H6("Q: What’s the main motivation of research behind FeedbackQA?"),
                        html.P("Users interact with QA systems and may leave feedback. "
                            "We investigate if this feedback is useful to improve QA systems in terms of accuracy and explainability. "
                            "And our research shows indeed that such feedback improves accuracy, "
                            "as well as it improves the ability of end-users to discern correct and incorrect answers."),
                        html.Hr(),
                        html.H6("Q: How does this demonstrated system work?"),
                        html.P("It contains a BERT-based retrieval model that fetches relevant passages for a given question, "
                            "and a BART-based Feedback reranker. "
                            "These models are trained separately with QA data and feedback data respectively from our FeedbackQA dataset. "
                            "Given a question, the RQA model pre-selects top-k candidates with highest relevant scores from a corpus of passages, "
                            "after which the Feedback reranker generates an explanation and a rating score for each candidate. "
                            "The rating scores and relevant scores are aggregated to rerank the candidates."),
                        html.Hr(),
                        html.H6("Q: Limitations of FeedbackQA?"),
                        html.P("(1) Although the dataset contains COVID-related content, "
                            "it is only for research purposes, and is not intended to provide medical advice. "
                            "We warn that some questions and answers could be emotionally disturbing."),
                        html.P("(2) Our data sources come from public health websites like WHO and CDC. "
                            "We do not collect any personal data.  "
                            "While we do not intend to reveal any personal information, if you come across any sensitive information, "
                            "please report back to us so we can update the data."),
                        html.P("(3) Neither the answers nor the explanations are always correct. We expect the models to make several mistakes.")
                        ]),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close", id="modal_close", className="ms-auto", n_clicks=0, color="info"
                        )
                    ),
                ],
                    id="modal",
                    is_open=False,
                    size="lg",
            #    )]
            ),
            #),
        ],
        #fluid=True,
        className="py-3",
    )

    search_bar = dbc.InputGroup(
        [
            # dbc.DropdownMenu(
            #     label="Examples",
            #     color="info",
            #     children=[
            #         dbc.DropdownMenuItem(p[:80] + "...", id={"type": "preset", "index": i})
            #         for i, p in enumerate(sample_questions)
            #     ],
            #     addon_type="prepend"
            # ),
            dbc.Input(id="search-bar", placeholder="Ask your question here..."),
            dbc.DropdownMenu(
                #nav=True,
                #in_navbar=True,
                label="# Answers",
                children=[
                    dbc.DropdownMenuItem(
                        f"{i} answers", id={"type": "n-answers", "index": i}
                    )
                    for i in range(1, 5)
                ],
                right=True,
                #color='light',
                toggle_style={"background": "white", "border": "#DCDCDC", "color": "#008b8b"},
                ),
            dbc.InputGroupAddon(
                #dcc.Loading([btn := html.Button('Submit'), output := html.Div()])
                dbc.Button("🔍 Search", id="search-button", color="info"),
                addon_type="append",
            ),
            dbc.InputGroupAddon(
                dbc.Button("✘ Clear", id="clear-button", color="danger"),
                addon_type="append",
            ),
            html.Div(id="log", children=[dcc.Store(id="trigger")], style={"display": "none"})
        ]
    )
    
    return dbc.Container(
        [
            navbar,
            html.Br(),
            intro,
            html.Br(),
            dcc.Dropdown(
                options=['WHO'],
                id="region",
                placeholder="Select a region...",
                value='WHO'
            ),
            dcc.Dropdown(
                options=[{"label": x, "value": x} for x in sample_questions[:15]],
                placeholder="Select a sample question...",
                id="preset",
            ),
            html.Br(),
            search_bar,
            html.Br(),
            html.Div([answer_component(n) for n in range(1, 4)], id="answers"),
            dcc.Store(id="n-answers-store", data=3),
        ],
        fluid=False,
    )


def assign_callbacks(app, system, sample_questions: List[str]) -> None:
    @app.callback(
        Output({"name": "collapse-preview", "index": MATCH}, "is_open"),
        Output({"name": "collapse-answer", "index": MATCH}, "is_open"),
        Output({"name": "show-more-btn", "index": MATCH}, "children"),
        Input({"name": "show-more-btn", "index": MATCH}, "n_clicks"),
    )
    def toggle_collapse(n):
        if not n:
            return [dash.no_update] * 3

        return n % 2 == 0, n % 2 != 0, "Show more" if n % 2 == 0 else "Show less"


    @app.callback(
        Output("modal", "is_open"),
        [Input("modal_open", "n_clicks"), Input("modal_close", "n_clicks")],
        [State("modal", "is_open")],
    )
    def toggle_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

    @app.callback(Output("answers", "children"), Input("n-answers-store", "data"))
    def change_n_cards(n):
        return [answer_component(n) for n in range(1, n + 1)]

    @app.callback(
        Output("n-answers-store", "data"),
        Input({"type": "n-answers", "index": ALL}, "n_clicks"),
    )
    def update_n_answers_store(n_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update

        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        ix = json.loads(prop_id)["index"]
        return ix

    @app.callback(
        dash.dependencies.Output('preset', 'options'),
        [dash.dependencies.Input('region', 'value')])
    def set_questions_options(region):
        random.shuffle(sample_questions)
        return [{"label": x, "value": x} for x in sample_questions[:15]]

    @app.callback(
        Output("search-bar", "value"),
        Output("search-button", "n_clicks"),
        Input("clear-button", "n_clicks"),
        # Input({"type": "preset", "index": ALL}, "n_clicks"),
        Input("preset", "value"),
        State("search-button", "n_clicks"),
    )
    def clear_search(n_clicks, preset, search_n_clicks):
        if search_n_clicks is None:
            search_n_clicks = 0

        ctx = dash.callback_context
        random.shuffle(sample_questions)
        if not ctx.triggered:
            return "", dash.no_update

        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "clear-button":
            return "", dash.no_update

        # selected_dict = json.loads(prop_id)
        # idx = selected_dict["index"]
        # return sample_questions[idx], search_n_clicks + 1
        return preset, search_n_clicks + 1

    @app.callback(
        Output("search-button", "disabled"), 
        Trigger("search-button", "n_clicks"), 
        Trigger("trigger", "data"))
    def disable_submit_button(n_clicks, trigger_data):
        context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        return context == 'search-button'

    @app.callback(
        Output("preset", "disabled"), 
        Trigger("search-button", "n_clicks"), 
        Trigger("trigger", "data"))
    def disable_preset(n_clicks, trigger_data):
        context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        return context == 'search-button'

    @app.callback(
        Output("clear-button", "disabled"), 
        Trigger("search-button", "n_clicks"), 
        Trigger("trigger", "data"))
    def disable_clear_button(n_clicks, trigger_data):
        context = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        return context == 'search-button'

    @app.callback(
        Output({"name": "show-more-btn", "index": ALL}, "disabled"),
        Output({"name": "markdown-preview", "index": ALL}, "children"),
        Output({"name": "markdown-answer", "index": ALL}, "children"),
        Output({"name": "markdown-feedback", "index": ALL}, "children"),
        Output({"name": "title-answer", "index": ALL}, "children"),
        Output({"name": "title-feedback", "index": ALL}, "children"),
        Output("log", "children"),
        Input("search-button", "n_clicks"),
        Input("search-bar", "n_submit"),
        State("region", "value"),
        State("search-bar", "value"),
        State("n-answers-store", "data"),
        prevent_initial_call=True,
    )
    def run_search_and_feedback(n_clicks, n_submit, region, query, n_answers):
        print('run_search_and_feedback')
        if n_answers is None:
            n_answers = 3
        if not query:
            return [dash.no_update] * 6 * n_answers

        time.sleep(0.75)

        while system.computing:
            time.sleep(1)

        system.computing = True
        best_passage, rating, feedback = system.retrieve_and_rerank(region, query)#[:n_answers]
        #best_passage = ['good answer'] * n_answers
        #rating = ['good'] * n_answers
        #feedback = ['good answer'] * n_answers

        best_passages_md = [convert_to_md(psge) for psge in best_passage[:n_answers]]
        #best_passages_content = [
            #p["content"] for p in best_passages
        #    best_passage
        #]  # TODO: change this

        #ratings = system.rate(query, best_passages_content)
        #feedback = system.give_feedback(query, best_passages_content)
        system.computing = False
        return [
            [len(c) <= 500 for c in best_passages_md],
            [c[:500] + "..." if len(c) > 500 else c for c in best_passages_md],
            best_passages_md,
            [f"Explanation: {x}" for x in feedback[:n_answers]],
            [f"Source: {region}" for _ in range(n_answers)],
            [f"Rating: {x}" for x in rating[:n_answers]],  # title-feedback
            ["Task completed", dcc.Store(id="trigger")],
        ]


def build(
    system,
    sample_questions: List[str],
    platform: str = "local",
    disconnect_tunnels: bool = True,
    **kwargs,
):
    platform = platform.lower()

    stylesheets = [dbc.themes.YETI, dbc.themes.BOOTSTRAP]
    cache = diskcache.Cache("./cache")
    long_callback_manager = DiskcacheLongCallbackManager(cache)
    if platform == "local":
        app = dash.Dash(__name__, external_stylesheets=stylesheets, prevent_initial_callbacks=True)

    elif platform == "colab":
        app = JupyterDash(__name__, external_stylesheets=stylesheets)

    elif platform == "kaggle":
        if disconnect_tunnels is True:
            tunnels = ngrok.get_tunnels()
            if len(tunnels) > 0:
                for tunnel in tunnels:
                    ngrok.disconnect(tunnel.public_url)
        
        port = kwargs.get("port", 80)

        tunnel = ngrok.connect(port)
        app = JupyterDash(
            __name__, external_stylesheets=stylesheets, server_url=tunnel.public_url
        )

    else:
        error_message = (
            f'platform="{platform}" is incorrect. Please choose between "local", "colab", '
            'and "kaggle".'
        )
        raise ValueError(error_message)

    app.layout = layout(sample_questions)
    assign_callbacks(app, system, sample_questions)

    return app, app.server


def start(system, sample_questions: List[str], platform: str = "local", **kwargs):
    app = build(system, sample_questions, platform, **kwargs)
    app.run_server(**kwargs)

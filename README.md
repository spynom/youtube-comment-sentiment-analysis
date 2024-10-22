# youtube comment sentiment analysis

<div>
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40" role="img" aria-label="CCDS: Project template">
    <title>CCDS: Project template</title>
    <defs>
        <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#2196F3"/>
            <stop offset="100%" stop-color="#03A9F4"/>
        </linearGradient>
        <clipPath id="r">
            <rect width="200" height="30" rx="5"/>
        </clipPath>
    </defs>
    <g clip-path="url(#r)">
        <rect width="200" height="40" fill="url(#bgGradient)"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="Verdana,sans-serif" font-size="20">
        <text x="20" y="21"> 🤗</text>
        <text x="100" y="21">CCDS</text>
    </g>
</svg>
</div>






this project on building a end-to-end a ml model building to model deployment to available to use for users.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         youtube_comment_sentiment_analysis and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── youtube_comment_sentiment_analysis   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes youtube_comment_sentiment_analysis a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


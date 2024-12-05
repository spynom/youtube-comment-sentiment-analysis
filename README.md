# Result:
### [demonstrate video](https://drive.usercontent.google.com/download?id=1c2eNer93hL0UoXi_cey6m5getU6FjnM7)
# Business Context:
We are "Influence Boost Inc.," an influencer management company seeking to expand our
network by attracting more influencers to join our platform. Due to a limited marketing
budget, traditional advertising channels are not viable for us. To overcome this, we aim to
offer a solution that addresses a significant pain point for influencers, thereby encouraging
them to engage with our company.
## Business Problem:
### 1. **Need to Attract More Influencers:**
**Objective:** Increase our influencer clientele to enhance our service offerings to
brands and stay competitive.<br>

**Challenge:** Limited marketing budget restricts our ability to reach and engage
potential influencer clients through conventional means.

### **2. Identifying the Influencer Pain Point:**

**Understanding Influencer Challenges:** To effectively attract influencers, we need to
understand and address the key challenges they face.

**Research Insight:** Influencers, especially those with large followings, struggle with
managing and interpreting the vast amount of feedback they receive via comments
on their content.

### **3. Big Influencers Face Issues with Comment Analysis:**

**Volume of Comments:** High-profile influencers receive thousands of comments on
their videos, making manual analysis impractical.

**Time Constraints:** Influencers often lack the time to sift through comments to extract
meaningful insights.

**Impact on Content Strategy:** Without efficient comment analysis, influencers miss
opportunities to understand audience sentiment, address concerns, and tailor their
content effectively.

# Our Solution
To directly address the significant pain point faced by big influencers—managing and
interpreting vast amounts of comment data—we present the "Influencer Insights" Chrome
plugin. This tool is designed to empower influencers by providing in-depth analysis of their
YouTube video comments, helping them make data-driven decisions to enhance their content
and engagement strategies.

## **Key Features of the Plugin:**

### **1. Sentiment Analysis of Comments**

* **Real-Time Sentiment Classification:**
The plugin performs real-time analysis of all comments on a YouTube video,
classifying each as positive, neutral, or negative.

* **Sentiment Distribution Visualization:**
Displays the overall sentiment distribution with intuitive graphs or charts (e.g., pie
charts or bar graphs showing percentages like 70% positive, 20% neutral, 10%
negative).

* **Detailed Sentiment Insights:**
Allows users to drill down into each sentiment category to read specific comments
classified under it.

* **Trend Tracking:**
Monitors how sentiment changes over time, helping influencers identify how
different content affects audience perception.

### **2. Additional Comment Analysis Features**

* **Word Cloud Visualization:**
Generates a word cloud showcasing the most frequently used words and phrases in
the comments.

* **Helps quickly identify trending topics, keywords, or recurring themes.**

* **Average Comment Length:**
Calculates and displays the average length of comments, indicating the depth of
audience engagement.

* **Spam and Troll Detection:**
Filters out spam, bot-generated comments, or potentially harmful content to
streamline the analysis.

# Workflow
**1. Data collection**<br>
**2. Data Preprocessing**<br>
**3. EDA**<br>
**4. Model building, Hyperparameter Tuning & Evaluation alongside Experiment Tracking**<br>
**5. Building a DVC pipeline**<br>
**6. Registering the model**<br>
**7. Building the API using Flask**<br>
**8. Development of Chrome Plugin**<br>
**9. Setting up CI/CD pipeline**<br>
**10. Testing**<br>
**11. Building the Docker image and pushing to hub**<br>
**12. Deployment using AWS**<br>

# Technologies

## **1. Version Control and Collaboration**

* ### Git 

  - **Purpose:** Distributed version control system for tracking changes in source code.<br>
  - **Usage:** Manage codebase, track changes, and collaborate with team members.

* ### GitHub

    - **Purpose:** Hosting service for Git repositories with collaboration features.<br>
    - **Usage:** Store repositories, manage issues, pull requests, and facilitate team
    collaboration.

## **2. Data Management and Versioning**

* ### DVC (Data Version Control)
    - **Purpose:** Version control system for tracking large datasets and machine learning
models.<br>
    - **Usage:** Version datasets and machine learning pipelines, enabling reproducibility and
collaboration.

* ### AWS S3 (Simple Storage Service)
    - **Purpose:** Scalable cloud storage service.<br>
    - **Usage:** Store datasets, pre-processed data, and model artifacts tracked by DVC.

## **3. Machine Learning and Experiment Tracking**

* ### Python
  - **Purpose**: Programming language for backend development and machine learning.
  - **Usage**: Implement data processing scripts, machine learning models, and backend services.

* ### Scikit-learn
  - **Purpose**: Library for classical machine learning algorithms.
  - **Usage**: Implement baseline models and preprocessing techniques.

* ### NLTK (Natural Language Toolkit)
  - **Purpose**: Platform for building Python programs to work with human language data.
  - **Usage**: Tokenization, stemming, and other basic NLP tasks.

* ### Mlflow
  - **Purpose**: Platform for managing the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry.
  - **Usage**: Track experiments, log parameters, metrics, and artifacts; manage model versions.

* ### MLflow Model Registry
  - **Purpose**: Component of MLflow for managing the full lifecycle of ML models.
  - **Usage**: Register models, manage model stages (e.g., staging, production), and collaborate on model development.

* ### Optuna
  - **Purpose**: Hyperparameter tuning.

## 4. Continuous Integration/Continuous Deployment (CI/CD)

* ### GitHub Actions
  - **Purpose**: Automation platform that enables CI/CD directly from GitHub repositories.
  - **Usage**:
    - Automate testing, building, and deployment pipelines.
    - Trigger workflows on events like code commits or pull requests.

## 5. Cloud Services and Infrastructure

* ### AWS (Amazon Web Services)

* ### AWS EC2 (Elastic Compute Cloud)
  - **Purpose**: Scalable virtual servers in the cloud.
  - **Usage**: Host backend services, APIs, and model servers.

* ### AWS Auto Scaling Groups
  - **Purpose**: Automatically adjust the number of EC2 instances to handle load changes.
  - **Usage**:
    - Scale in during low demand periods to reduce costs.
    - Maintain application availability by automatically adding or replacing instances as needed.

* ### AWS CodeDeploy
  - **Purpose**: Deployment service that automates application deployments to various compute services like EC2, Lambda, and on-premises servers.
  - **Usage**:
    - Automate the deployment process of backend services and machine learning models to AWS EC2 instances or AWS Lambda.
    - Integrate with GitHub Actions to create a seamless CI/CD pipeline that deploys code changes automatically upon successful testing.

* ### AWS CloudWatch
  - **Purpose**: Monitoring and observability service.
  - **Usage**: Monitor application logs, set up alerts, and track performance metrics.

* ### AWS IAM (Identity and Access Management)
  - **Purpose**: Securely manage access to AWS services.
  - **Usage**: Control access permissions for users and services.

## 6. Testing and Quality Assurance Tools

* ### Unittest
  - **Purpose**: Built-in Python testing framework.
  - **Usage**: Write unit tests for Python code.

## 7. API Development and Testing

* ### Frameworks:

* ### FastAPI
  - **Purpose**: Modern, fast web framework for building APIs with Python.
  - **Usage**: Develop high-performance APIs efficiently.

* ### API Testing Tools:

* ### Postman
  - **Purpose**: API development environment.
  - **Usage**: Design, test, and document APIs.

## 8. Frontend Development Tools

* ### Chrome Extension APIs
  - **Purpose**: APIs provided by Chrome for building extensions.
  - **Usage**: Interact with browser features, modify web page content, manage extension behavior.

* ### Browser Developer Tools
  - **Purpose**: Built-in tools for debugging and testing web applications.
  - **Usage**: Inspect elements, debug JavaScript, monitor network activity.

* ### Code Editors and IDEs:

* ### Visual Studio Code
  - **Purpose**: Source code editor.
  - **Usage**: Write and edit code for both frontend and backend development.

## 9. Additional Tools and Libraries

* ### Matplotlib
  - **Purpose**: Plotting library for Python.
  - **Usage**: Create static, animated, and interactive visualizations.

* ### Seaborn
  - **Purpose**: Statistical data visualization.
  - **Usage**: Generate high-level interface for drawing attractive graphics.

* ### D3.js
  - **Purpose**: JavaScript library for producing dynamic, interactive data visualizations.
  - **Usage**: Create word clouds and other visual elements in the Chrome extension.

* ### Data Serialization Formats:

* ### JSON
  - **Purpose**: Lightweight data interchange format.
  - **Usage**: Transfer data between frontend and backend services.

* ### Docker
  - **Purpose**: Containerization platform.
  - **Usage**: Package applications and dependencies into containers for consistent
deployment.

# Experiments
In this project, we conducted several experiments to evaluate the performance of the model on different algo with hyperparameters and options.

## 1. Base Model Metrics
| Model | Accuracy | Class 0 Precision | Class 1 Recall | Class 1 Precision | Class 1 Recall | Class 2 Precision | Class 2 Recall |
|---|----------|-------------------|----|----|----|----|----|
|RandomForestClassifier| 0.658 | 1.000 | 0.000 |0.670 | 0.81 | 0.62 | 0.84 | 0.71 |

## 2. TF-IDF VS BOW
| Hyperparameter                | Model 1 | Model 2 | Model 3 | Model 4 | Model 5 | Model 6 |
|-------------------------------|---------|---------|---------|---------|---------|---------|
| **max_depth**                  | 15      | 15      | 15      | 15      | 15      | 15      |
| **n_estimators**               | 200     | 200     | 200     | 200     | 200     | 200     |
| **ngram_range**                | (1, 3)  | (1, 3)  | (1, 2)  | (1, 2)  | (1, 1)  | (1, 1)  |
| **vectorizer_max_features**    | 5000    | 5000    | 5000    | 5000    | 5000    | 5000    |
| **vectorizer_type**            | TF-IDF  | BoW     | TF-IDF  | BoW     | TF-IDF  | BoW     |

### Metrics

| Metric                        | Model 1 | Model 2 | Model 3 | Model 4 | Model 5 | Model 6 |
|-------------------------------|---------|---------|---------|---------|---------|---------|
| **0_f1-score**                | 0.071   | 0.071   | 0.065   | 0.065   | 0.046   | 0.046   |
| **0_precision**               | 0.968   | 0.968   | 0.982   | 0.982   | 1.000   | 1.000   |
| **0_recall**                  | 0.037   | 0.037   | 0.034   | 0.034   | 0.024   | 0.024   |
| **0_support**                 | 1640    | 1640    | 1640    | 1640    | 1640    | 1640    |
| **1_f1-score**                | 0.744   | 0.744   | 0.745   | 0.745   | 0.736   | 0.736   |
| **1_precision**               | 0.679   | 0.679   | 0.685   | 0.685   | 0.671   | 0.671   |
| **1_recall**                  | 0.824   | 0.824   | 0.816   | 0.816   | 0.816   | 0.816   |
| **1_support**                 | 2475    | 2475    | 2475    | 2475    | 2475    | 2475    |
| **2_f1-score**                | 0.725   | 0.725   | 0.725   | 0.725   | 0.720   | 0.720   |
| **2_precision**               | 0.634   | 0.634   | 0.630   | 0.630   | 0.629   | 0.629   |
| **2_recall**                  | 0.846   | 0.846   | 0.853   | 0.853   | 0.843   | 0.843   |
| **2_support**                 | 3133    | 3133    | 3133    | 3133    | 3133    | 3133    |
| **accuracy**                  | 0.655   | 0.655   | 0.655   | 0.655   | 0.648   | 0.648   |


## 3. TF-IDF with different no. of max features

| Hyperparameter                | Model 1 | Model 2 | Model 3 | Model 4 | Model 5 | Model 6 | Model 7 | Model 8 | Model 9 | Model 10 |
|-------------------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|
| **max_depth**                  | 15      | 15      | 15      | 15      | 15      | 15      | 15      | 15      | 15      | 15       |
| **n_estimators**               | 200     | 200     | 200     | 200     | 200     | 200     | 200     | 200     | 200     | 200      |
| **ngram_range**                | (1, 3)  | (1, 3)  | (1, 3)  | (1, 3)  | (1, 3)  | (1, 3)  | (1, 3)  | (1, 3)  | (1, 3)  | (1, 3)   |
| **vectorizer_max_features**    | 10000   | 9000    | 8000    | 7000    | 6000    | 5000    | 4000    | 3000    | 2000    | 1000     |
| **vectorizer_type**            | TF-IDF  | TF-IDF  | TF-IDF  | TF-IDF  | TF-IDF  | TF-IDF  | TF-IDF  | TF-IDF  | TF-IDF  | TF-IDF   |

### Metrics

| Metric                        | Model 1 | Model 2 | Model 3 | Model 4 | Model 5 | Model 6 | Model 7 | Model 8 | Model 9 | Model 10 |
|-------------------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|
| **0_f1-score**                | 0.018   | 0.042   | 0.041   | 0.055   | 0.045   | 0.071   | 0.067   | 0.080   | 0.163   | 0.234    |
| **0_precision**               | 1.000   | 1.000   | 1.000   | 1.000   | 1.000   | 0.968   | 0.966   | 0.986   | 0.925   | 0.891    |
| **0_recall**                  | 0.009   | 0.021   | 0.021   | 0.028   | 0.023   | 0.037   | 0.035   | 0.041   | 0.090   | 0.135    |
| **0_support**                 | 1640    | 1640    | 1640    | 1640    | 1640    | 1640    | 1640    | 1640    | 1640    | 1640     |
| **1_f1-score**                | 0.738   | 0.746   | 0.744   | 0.740   | 0.732   | 0.744   | 0.732   | 0.738   | 0.736   | 0.734    |
| **1_precision**               | 0.713   | 0.704   | 0.696   | 0.696   | 0.671   | 0.679   | 0.661   | 0.658   | 0.645   | 0.627    |
| **1_recall**                  | 0.764   | 0.794   | 0.799   | 0.791   | 0.806   | 0.824   | 0.821   | 0.840   | 0.857   | 0.885    |
| **1_support**                 | 2475    | 2475    | 2475    | 2475    | 2475    | 2475    | 2475    | 2475    | 2475    | 2475     |
| **2_f1-score**                | 0.717   | 0.724   | 0.723   | 0.721   | 0.718   | 0.725   | 0.718   | 0.723   | 0.722   | 0.724    |
| **2_precision**               | 0.603   | 0.618   | 0.621   | 0.618   | 0.625   | 0.634   | 0.633   | 0.643   | 0.658   | 0.685    |
| **2_recall**                  | 0.882   | 0.873   | 0.867   | 0.865   | 0.844   | 0.846   | 0.831   | 0.824   | 0.799   | 0.767    |
| **2_support**                 | 3133    | 3133    | 3133    | 3133    | 3133    | 3133    | 3133    | 3133    | 3133    | 3133     |
| **accuracy**                  | 0.644   | 0.653   | 0.652   | 0.650   | 0.646   | 0.655   | 0.647   | 0.653   | 0.658   | 0.664    |

## 3. imbalance handling methods 
| Hyperparameter                | SMOTE_ENN | Undersampling | ADASYN  | Oversampling | Class_Weights |
|-------------------------------|-----------|----------------|---------|---------------|---------------|
| **max_depth**                  | 15        | 15             | 15      | 15            | 15            |
| **n_estimators**               | 200       | 200            | 200     | 200           | 200           |
| **ngram_range**                | (1, 3)    | (1, 3)         | (1, 3)  | (1, 3)        | (1, 3)        |
| **vectorizer_max_features**    | 1000      | 1000           | 1000    | 1000          | 1000          |
| **vectorizer_type**            | TF-IDF    | TF-IDF         | TF-IDF  | TF-IDF        | TF-IDF        |

### Metrics

| Metric                        | SMOTE_ENN | Undersampling | ADASYN  | Oversampling | Class_Weights |
|-------------------------------|-----------|----------------|---------|---------------|---------------|
| **0_f1-score**                | 0.436     | 0.510          | 0.493   | 0.491         | 0.498         |
| **0_precision**               | 0.291     | 0.551          | 0.570   | 0.558         | 0.580         |
| **0_recall**                  | 0.870     | 0.475          | 0.435   | 0.439         | 0.437         |
| **0_support**                 | 1309      | 1309           | 1309    | 1309          | 1309          |
| **1_f1-score**                | 0.632     | 0.726          | 0.729   | 0.725         | 0.729         |
| **1_precision**               | 0.658     | 0.604          | 0.603   | 0.603         | 0.605         |
| **1_recall**                  | 0.608     | 0.910          | 0.920   | 0.909         | 0.916         |
| **1_support**                 | 1997      | 1997           | 1997    | 1997          | 1997          |
| **2_f1-score**                | 0.032     | 0.675          | 0.691   | 0.689         | 0.692         |
| **2_precision**               | 1.000     | 0.844          | 0.836   | 0.833         | 0.828         |
| **2_recall**                  | 0.016     | 0.563          | 0.588   | 0.587         | 0.595         |
| **2_support**                 | 2493      | 2493           | 2493    | 2493          | 2493          |
| **accuracy**                  | 0.413     | 0.663          | 0.668   | 0.665         | 0.670         |


## 4. Different classification algo.

| **Metric**               | **XGBoost** | **SVM** | **MultinomialNB** | **LightGBM** | **KNN** |
|--------------------------|-------------|---------|-------------------|--------------|---------|
| **0_f1-score**           | 0.674       | 0.760   | 0.506             | 0.718        | 0.100   |
| **0_precision**          | 0.845       | 0.744   | 0.739             | 0.746        | 0.704   |
| **0_recall**             | 0.561       | 0.776   | 0.384             | 0.693        | 0.054   |
| **0_support**            | 1640        | 1640    | 1640              | 1640         | 1640    |
| **1_f1-score**           | 0.843       | 0.902   | 0.638             | 0.848        | 0.525   |
| **1_precision**          | 0.752       | 0.855   | 0.830             | 0.755        | 0.357   |
| **1_recall**             | 0.958       | 0.955   | 0.519             | 0.967        | 0.990   |
| **1_support**            | 2475        | 2475    | 2475              | 2475         | 2475    |
| **2_f1-score**           | 0.826       | 0.860   | 0.714             | 0.814        | 0.143   |
| **2_precision**          | 0.844       | 0.916   | 0.587             | 0.907        | 0.931   |
| **2_recall**             | 0.809       | 0.811   | 0.909             | 0.739        | 0.077   |
| **2_support**            | 3133        | 3133    | 3133              | 3133         | 3133    |
| **test_accuracy**        | 0.804       | 0.852   | 0.657             | 0.806        | 0.384   |
| **train_accuracy**       | 0.865       | 0.933   | 0.797             | 0.852        | 0.426   |

## 5. SVC and LightBGM with the best hyperparameters after 100 trails
| **Hyperparameter**           | **LightGBM**                              |
|------------------------------|-------------------------------------------|
| **colsample_bytree**          | 0.8172519668728906                       |
| **learning_rate**             | 0.08642877409476625                      |
| **max_depth**                 | 4                                         |
| **min_child_samples**         | 22                                        |
| **n_estimators**              | 885                                       |
| **num_leaves**                | 99                                        |
| **reg_alpha**                 | 0.0003432471870865868                    |
| **reg_lambda**                | 0.4933542438262955                       |
| **subsample**                 | 0.9248744925482706                       |

| **Hyperparameter**           | **SVC**                                   |
|------------------------------|-------------------------------------------|
| **kernel**                    | linear                                    |
| **shrinking**                 | False                                     |


| **Metric**               | **LightGBM** | **SVC** |
|--------------------------|--------------|---------|
| **0_f1-score**           | 0.774        | 0.761   |
| **0_precision**          | 0.794        | 0.745   |
| **0_recall**             | 0.754        | 0.777   |
| **0_support**            | 1640         | 1640    |
| **1_f1-score**           | 0.881        | 0.904   |
| **1_precision**          | 0.805        | 0.856   |
| **1_recall**             | 0.974        | 0.957   |
| **1_support**            | 2475         | 2475    |
| **2_f1-score**           | 0.853        | 0.861   |
| **2_precision**          | 0.922        | 0.917   |
| **2_recall**             | 0.793        | 0.811   |
| **2_support**            | 3133         | 3133    |
| **test_accuracy**        | 0.872        | 0.85    |
| **train_accuracy**       | 0.885        | 0.92    |


# Project Organization

```

├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── raw            <- Raw Data from third party sources.
│   ├── Processed      <- Processed data that has been cleaned and pre-processed.
│   ├── final          <- The final, canonical data sets for modeling.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks well explained related to data, preprocess idea, EDA and Experiments.
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
├── scripts            <- Scripts for test and promote model
│
├── app                <- Source files for application
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes youtube_comment_sentiment_analysis a Python module
    │
    ├── setup_logger.py         <- Script to setup logger for scripts
    │
    ├── data_ingestion.py       <- Scripts to download or generate data
    │
    ├── data_preprocessing.py   <- Scripts to clean the generated raw data
    │
    ├── feature_engineering.py  <- Code to do feature_engineering on data for modeling
    │
    ├── model_building.py       <- Script to train the model and save it
    │
    ├── model_evaluation.py     <- to evaluate and log metrices of the trained model on given test data
    │
    └── model_registry.py       <- Script to register the current model on the registry.

```

--------


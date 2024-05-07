MLflow Pipeline
==============================

Built E2E ML Pipeline with MLflow & AWS.

## Workflow
<p align = "center">
  <img class="center" src = "https://github.com/ronylpatil/mlflow-pipeline/blob/aws/workflow/flow.png" alt = "Drawing">
</p>

## Blogs

- Part I - [Streamline ML Workflow with MLflow - I](https://medium.com/towards-artificial-intelligence/streamline-ml-workflow-with-mlflow%EF%B8%8F-part-i-60857cd511ed)
- Part II - [Streamline ML Workflow with MLflow - II](https://medium.com/towards-artificial-intelligence/streamline-ml-workflow-with-mlflow-ii-daa8d50016f7)
- Part III - [Configure DVC with Amazon S3 Bucket](https://medium.com/towards-artificial-intelligence/configure-dvc-with-amazon-s3-bucket-f6d57cd242d4)
- Part IV - [Deploy MLflow Server on EC2 Instance](https://medium.com/towards-artificial-intelligence/deploy-mlflow-server-on-amazon-ec2-instance-b53d5eb3c4f3)

## Installation

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`

## Folder Structure
- `/.github`: Contains CI/CD workflow file.
- `/.dvc`: Contains configuration files of DVC
- `/data`: Stores raw and processed data.
- `/log`: Store the logs.
- `/src`: Contains the source code files.
- `/prod`: Production files.
- `/tests`: Testing files.

## Dataset

- Download the dataset from [here](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset).

---
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. </small></p>

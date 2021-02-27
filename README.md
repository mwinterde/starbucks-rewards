# Starbucks Mobile Reward Program Challenge

## Project Overview

In this project we work with simulated data that mimics customer behavior on the Starbucks rewards mobile app. Based on various user demographics and offer characteristics, we want to build a recommendation engine that allows marketing managers to choose the best offer for a customer, that is, it provides the highest probability of completion. Therefore we train and deploy a classification model using Amazon SageMaker which provides the completion probability for a given customer-offer combination. These predictions can then be used to recommend the best offer for a given user to the marketing team. 


## Installation

### Start AWS SageMaker Notebook Instance

Start a AWS SageMaker notebook instance  and launch a terminal to clone this repository to the `~/SageMaker` directory. 

```
git clone https://github.com/mwinterde/udacity-nd009t-capstone.git ~/SageMaker/udacity-nd009t-capstone
```

### Install Dependencies

If you are working on a SageMaker notebook instance, there is nothing to do for you. In this project, we will only work with standard libraries that are already part of the pre-installed notebook kernels. If you are working on your local machine use conda to install the dependencies:

Create an environment:

```
conda env create --file environment.yml
```

Activate the environment:

```
conda activate udacity-nd009t-capstone
```


## File Descriptions

### Notebooks

* `01_Data_Exploration_And_Preprocessing.ipynb` - first exploration of the given datasets + preparation of the data for model training
* `02_Broad_Model_Comparison.ipynb` - broad comparison between the predictive power of Logistic Regression, KNN and Random Forest models
* `03_Model_Deployment_And_Application.ipynb` - final training and deployment of a Random Forest model + illustration of the major use case


### Data

The data is contained in three files:

* `portfolio.json` - containing offer ids and meta data about each offer (duration, type, etc.)
* `profile.json` - demographic data for each customer
* `transcript.json` - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record


## Acknowledgements

This project was completed as part of the [Udacity Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t). 
The dataset of this project simulates customer behavior on the Starbucks rewards mobile app. [Starbucks® Rewards program: Starbucks Coffee Company](https://www.starbucks.com/rewards/).

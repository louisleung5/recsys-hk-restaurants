# recsys-hk-restaurants

## Recommendation System using Matrix Factorization in PyTorch

## Project Overview
This project implements a recommendation system using the matrix factorization technique. The machine learning model is built using the PyTorch library and is trained on a dataset containing ratings of restaurants given by customers in Hong Kong.

### Features
Matrix Factorization Technique: The recommendation system is built using the matrix factorization technique, which is a popular approach in collaborative filtering for recommendation systems. This method learns latent features from the user-item interaction data and uses them to predict new recommendations.
PyTorch Implementation: The machine learning model is implemented using the PyTorch library, a powerful open-source machine learning framework. PyTorch provides a flexible and efficient platform for building and training neural networks, which is well-suited for the matrix factorization model used in this project.
Hong Kong Restaurant Ratings Dataset: The project utilizes a dataset that contains ratings of restaurants given by customers in Hong Kong. This dataset provides the necessary user-item interaction data to train the recommendation model.

### Installation and Usage
To use this recommendation system, follow these steps:

1. Clone the repository to your local machine:
`git clone https://github.com/your-username/your-repo-name.git`

2. Install the required dependencies, including PyTorch and any other necessary libraries:
`pip install -r requirements.txt`

3. Prepare the dataset:
- Download the Hong Kong restaurant ratings dataset and place it in the appropriate directory in your project.
- Ensure that the dataset is in a compatible format for the recommendation system.

4. Run the recommendation system:
Execute the main script to train the matrix factorization model and generate recommendations. The script will output a list of recommended restaurants for each customer included in the dataset.

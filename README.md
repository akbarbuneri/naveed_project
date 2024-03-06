
# Flask Application for Product Similarity Search

This Flask application allows for the uploading of product details via an Excel file, generates embeddings for these products using the OpenAI API, and uses FAISS to enable efficient similarity searches among the products. It supports adding individual products through a JSON API, uploading product lists via Excel files, and querying similar products based on textual descriptions.

## Features

- **File Upload**: Accept Excel files containing product details and automatically process them to generate embeddings.
- **Add Product**: Add individual product details through a JSON API endpoint.
- **Similarity Search**: Query for similar products based on a textual description, utilizing generated embeddings and FAISS for efficient search.

## Requirements

- Python 3.8+
- Flask
- Pandas
- NumPy
- FAISS
- OpenAI
- python-dotenv

## Setup and Installation

1. **Clone the repository:**

    ```
    git clone https://github.com/akbarbuneri/naveed_project
    cd naveed_project
    ```

2. **Install dependencies:**

    Create a virtual environment and activate it:

    ```
    python -m venv nproj
    source nproj/bin/activate  # On Windows use `nproj\Scripts\activate`
    ```

    Install required Python packages:

    ```
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**

    Copy the `.env.example` file to `.env` and fill in your OpenAI API key:

    ```
    OPENAI_API_KEY=<your-api-key>
    ```

4. **Run the application:**

    ```
    flask run 
    ```
    Or if you want to debug
    ```
    flask --app app --debug run
    ```

## Usage

### 1. Adding a Product

- **Endpoint**: `/add_product`
- **Method**: POST
- **Data**:

    ```json
    {
      "ItemID": "123",
      "ItemName": "Example Product",
      "Brand": "Example Brand",
      "Price": "999",
      "Description": "Example description of the product."
    }
    ```

### 2. Uploading an Excel File

- **Endpoint**: `/upload`
- **Method**: POST
- **Form Data**: `file`: (Attach the Excel file here)

The Excel file should have columns: `ItemID`, `ItemName`, `Brand`, `Price`, `Description`.

### 3. Searching for Similar Products

- **Endpoint**: `/search?query=<search-text>`
- **Method**: GET

Returns a list of similar products based on the query.

## Structure

- **`app.py`**: Main application file with Flask routes.
- **`storage/`**: Directory to store FAISS index and product IDs persistently.
- **`uploaded/`**: Temporary storage for uploaded Excel files.

## Notes

- Ensure the `UPLOAD_FOLDER` and `storage/` directories exist and are writable.
- The similarity calculation in the search functionality is based on inverse L2 distance; you may adjust this according to your requirements.


# Similarity Measures Examples

When calculating similarity from distance metrics, several approaches can be used depending on the characteristics of your data and the requirements of your application. Here are some commonly used similarity measures and their potential applications:

## Cosine Similarity

Used in text analysis and information retrieval, measuring the cosine of the angle between two vectors.

```python
from sklearn.metrics.pairwise import cosine_similarity
# Assuming A and B are your vectors
similarity = cosine_similarity(A.reshape(1, -1), B.reshape(1, -1))
```

## Jaccard Similarity

Useful for comparing the similarity and diversity of sample sets.

```python
def jaccard_similarity(set_a, set_b):
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union
```

## Euclidean Distance to Similarity Conversion

Transforming Euclidean distance into a similarity score.

```python
import numpy as np
def euclidean_similarity(vec_a, vec_b):
    distance = np.linalg.norm(vec_a - vec_b)
    similarity = 1 / (1 + distance)
    return similarity
```

## Pearson Correlation Coefficient

Measures the linear correlation between two datasets, interpretable as a similarity measure in many contexts.

```python
from scipy.stats import pearsonr
# Assuming A and B are your datasets
correlation, _ = pearsonr(A, B)
```

## Adjusted Cosine Similarity

Often used in collaborative filtering, adjusting for different rating scales.

```python
# This example is conceptual and would need to be adapted to your specific data structure
def adjusted_cosine_similarity(ratings_a, ratings_b, avg_ratings):
    adjusted_a = ratings_a - avg_ratings
    adjusted_b = ratings_b - avg_ratings
    similarity = np.dot(adjusted_a, adjusted_b) / (np.linalg.norm(adjusted_a) * np.linalg.norm(adjusted_b))
    return similarity
```

## Dot Product

For non-normalized vectors, emphasizing items with higher values or popularity.

```python
def dot_product_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b)
```

# Code Changes for the Search Method

To incorporate different similarity measures into the search method of your Flask application, you might adjust the `/search` endpoint. Below is an example modification to use cosine similarity (assuming embeddings are normalized):

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    query_embedding = generate_embedding(query).reshape(1, -1)  # Ensure query_embedding is correctly shaped
    top_k = 10
    
    # Perform the search with FAISS
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    # Retrieve the actual vectors for the returned indices
    closest_vectors = np.array([faiss_index.reconstruct(int(idx)) for idx in indices[0]])
    
    # Compute cosine similarity between the query and the closest vectors
    cos_similarities = cosine_similarity(query_embedding, closest_vectors)[0]
    
    # Prepare and return the results
    results = [{"ItemID": product_ids[int(indices[0][i])], "Similarity": cos_similarities[i]} for i in range(top_k)]
    
    return jsonify(results), 200
```

This code snippet assumes you have a preloaded array `all_embeddings` of embeddings for all items, and `product_ids` is a list that maps indices to ItemIDs.

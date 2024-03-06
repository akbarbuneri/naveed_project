
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
    git clone <repository-url>
    cd <repository-folder>
    ```

2. **Install dependencies:**

    Create a virtual environment and activate it:

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
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

## Usage

### Adding a Product

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

### Uploading an Excel File

- **Endpoint**: `/upload`
- **Method**: POST
- **Form Data**: `file`: (Attach the Excel file here)

The Excel file should have columns: `ItemID`, `ItemName`, `Brand`, `Price`, `Description`.

### Searching for Similar Products

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

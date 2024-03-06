from werkzeug.utils import secure_filename
import os
import pandas as pd

from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
import faiss
import numpy as np
from flask import Flask, request, jsonify,flash, redirect
import pickle  
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = 'uploaded'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Initialize a FAISS index
dimension = 1536    # Dimension of the embeddings; adjust based on the model used
faiss_index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity search
product_ids = []  # To keep track of product IDs corresponding to FAISS index entries

index_file = "storage/faiss_index.index"
if os.path.exists(index_file):
    faiss_index = faiss.read_index(index_file)
else:
    dimension = 1536  
    faiss_index = faiss.IndexFlatL2(dimension)

# Load or initialize the product IDs list
ids_file = "storage/product_ids.pkl"
if os.path.exists(ids_file):
    with open(ids_file, 'rb') as f:
        product_ids = pickle.load(f)
else:
    product_ids = []

def generate_embedding(text):
    response = client.embeddings.create(input=[text],
    model="text-embedding-ada-002")
    embedding = response.data[0].embedding
    return np.array(embedding, dtype='float32').reshape(1, -1)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        process_file(filepath)  # Process the file to generate and store embeddings
        return jsonify({"message": "File uploaded and processed successfully"}), 200

def process_file(filepath):
    df = pd.read_excel(filepath)
    # Here you would loop through the dataframe and generate & store embeddings
    for index, row in df.iterrows():
        item_id = row['ItemID']
        item_name = row.get('ItemName', '')
        brand = row.get('Brand', '')
        price = row.get('Price', '')
        description = row.get('Description', '')
        
        embedding_text = f'Item ID:{item_id}, Item Name:"{item_name}", Brand:"{brand}", Price:{price}, Description:"{description}"'
        embedding = generate_embedding(embedding_text)
        
        # Assuming item_id is unique and can be used to track embeddings
        if item_id not in product_ids:
            product_ids.append(item_id)
            faiss_index.add(embedding.reshape(-1, dimension))
            # Optionally, store item details in another structure
        
        # Save changes to disk after processing each item or batch
        faiss.write_index(faiss_index, index_file)
        with open(ids_file, 'wb') as f:
            pickle.dump(product_ids, f)
@app.route('/add_product', methods=['POST'])
def add_product():
    product_data = request.json
    item_id = product_data.get('ItemID')  # Use 'ItemID' directly from the product data
    
    # Ensure 'ItemID' is provided and is unique
    if item_id is None:
        return jsonify({"message": "ItemID is missing"}), 400
    if item_id in product_ids:
        return jsonify({"message": "This ItemID already exists"}), 400

    product_ids.append(item_id)  # Add 'ItemID' to 'product_ids'
    
    # Generate embedding
    if 'ItemName' in product_data:
        description = product_data.get('description', '')  # Ensure description is a string
        embedding_text = f'Item ID:{item_id}, Item Name:"{product_data["ItemName"]}", Brand:"{product_data["Brand"]}", Price:{product_data["Price"]}, Description:"{description}"'
        embedding = generate_embedding(embedding_text)
        faiss_index.add(embedding)
        # Save changes to disk
        faiss.write_index(faiss_index, index_file)
        with open(ids_file, 'wb') as f:
            pickle.dump(product_ids, f)
    
    return jsonify({"message": "Product added successfully", "ItemID": item_id}), 200

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    query_embedding = generate_embedding(query)
    # Search the FAISS index for the top 10 most similar embeddings
    top_k = 10
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    # Process results to match desired output format
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0:  # Check if the index is valid (FAISS can return -1 for some queries)
            item_id = product_ids[idx]
            # Convert distance to similarity for demonstration purposes. 
            # This is a simplistic conversion and may need adjustment based on your requirements.
            similarity = 1 / (1 + distances[0][i])  # Example conversion, adjust as needed
            results.append({"ItemID": item_id, "Similarity": similarity})
    
    return jsonify(results), 200

if __name__ == '__main__':
    app.run(debug=True)

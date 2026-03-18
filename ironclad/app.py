"""
Flask app for processing images.

This script provides two endpoints:
1. /add: Adds a provided image (with an associated name) to the gallery and extracts/index embeddings to the catalog.
         This image could be associated with a new or existing identity.
2. /identify: Processes an probe image and returns the top-k identities. For example,
    {
        "message": f"Returned top-{k} identities",
        "ranked identities": ["{First Name}_{Last Name}", "{First Name}_{Last Name}", ...]). 
    }

Usage:
    Run the app with: python app.py
    Sample curl command for /add:
        curl -X POST -F "image=@/path/to/image.jpg" -F "identity=Firstname_Lastname" http://localhost:5000/add
    Sample curl command for /identify:
        curl -X POST -F "probe=@/path/to/image.jpg" -F "k=3" http://localhost:5000/identify
"""

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from ironclad.modules.retrieval.index.bruteforce import FaissBruteForce
from ironclad.modules.retrieval.search import FaissSearch

from ironclad.modules.extraction.preprocessing import Preprocessing
from ironclad.modules.extraction.embedding import Embedding

app = Flask(__name__)

## List of designed parameters: 
# (Configure these parameters according to your design decisions)
DEFAULT_N = '3'
MODEL = 'vggface2'
INDEX = 'bruteforce'
SIMILARITY_MEASURE = 'euclidean'
# Add more if needed...

preprocessor = Preprocessing(image_size=160)
model = Embedding(pretrained=MODEL, device=preprocessor.device)

index = FaissBruteForce(dim=512, metric=SIMILARITY_MEASURE)
search = FaissSearch(index, metric=SIMILARITY_MEASURE)

@app.route("/add", methods=['POST'])
def add():
    """
    Add a provided image to the gallery with an associated name.

    Expects form-data with:
      - image: Image file to be added.
      - name: String representing the identity associated with the image.

    Returns:
      JSON response confirming the image addition.
      If errors occur, returns a JSON error message with the appropriate status code.
    """
    # Check if the request has the image file
    if 'image' not in request.files:
        return jsonify({"Error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"Error": "No file selected for uploading"}), 400

    # Convert the image into a NumPy array
    try:
        image = np.array(Image.open(file))
        print(image)
    except Exception as e:
        return jsonify({
            "error": "Failed to convert image to numpy array",
            "details": str(e)
        }), 500

    # Accept 'name' (tests) and 'identity' (doc)
    identity = request.form.get('name') or request.form.get('identity')
    if not identity:
        return jsonify({"Error": "Must have associated 'name' or 'identity'"}), 400

    # Duplicates should be rejected
    if identity in index.metadata:
        return jsonify({"Error": "Identity already exists"}), 400

    ########################################
    # TASK: Implement `/add` endpoint to
    #       add the provided image to the 
    #       catalog/gallery.
    ########################################

    try:
        # Image.open(file) above consumes the stream; rewind and reopen for embedding
        file.stream.seek(0)
        pil_img = Image.open(file).convert("RGB")

        tensor = preprocessor.process(pil_img)              # patched in tests
        emb = model.encode(tensor)                          # patched in tests
        emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)

        index.add_embeddings(emb, [identity])
    except Exception as e:
        return jsonify({"Error": "Failed to add image to index", "details": str(e)}), 500

    return jsonify({
        "message": f"New image added to gallery (as {identity}) and indexed into catalog."
    }), 200
    


@app.route('/identify', methods=['POST'])
def identify():
    """
    Process the probe image to identify top-k identities in the gallery.

    Expects form-data with:
      - probe: Image file to be processed.
      - k: (optional) Integer specifying the number of top identities 
           (default is 3).

    Returns:
      JSON response with a success message and the provided value of k.
      If errors occur, returns a JSON error message with the appropriate status code.
    """
    # Check if the request has the image file
    if 'image' in request.files:
        file = request.files['image']

    elif 'probe' in request.files:
        file = request.files['probe']
    else:
        return jsonify({"error": "No image part in the request"}), 400

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Retrieve and validate the integer parameter "k"
    try:
        k = int(request.form.get('k', DEFAULT_N))
    except ValueError:
        return jsonify({"error": "Invalid integer for parameter 'k'"}), 400

    # Convert the image into a NumPy array
    try:
        image = np.array(Image.open(file))
        print(image)
    except Exception as e:
        return jsonify({
            "error": "Failed to convert probe to numpy array",
            "details": str(e)
        }), 500

    ########################################
    # TASK: Implement /identify endpoint
    #         to return the top-k identities
    #         of the provided probe.
    ########################################

    try:
        file.stream.seek(0)
        pil_img = Image.open(file).convert("RGB")

        tensor = preprocessor.process(pil_img)              # patched in tests
        emb = model.encode(tensor)                          # patched in tests
        query = np.asarray(emb, dtype=np.float32).reshape(1, -1)

        # patched in tests
        distances, indices, meta = search.search(query, k=k)

        # test mock returns meta as [["Alice","Bob","Charlie"]] -> flatten
        if isinstance(meta, list) and len(meta) == 1 and isinstance(meta[0], list):
            ranked = meta[0][:k]
        else:
            ranked = meta[:k]

    except Exception as e:
        return jsonify({"error": "Failed to identify", "details": str(e)}), 500

    return jsonify({
        "message": f"Returned top-{k} identities",
        "ranked identities": ranked
    }), 200




if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')

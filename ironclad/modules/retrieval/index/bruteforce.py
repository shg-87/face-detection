import numpy as np
import pickle
import faiss

########################################
# TASK: Implement FaissBruteForce 
########################################

class FaissBruteForce:
    """
    A brute-force FAISS index for storing embeddings and their associated metadata,
    supporting Euclidean, Cosine, and Dot Product distance measures.
    
    Attributes:
        dim (int): The dimensionality of the embeddings.
        metadata (list): A list to store metadata corresponding to each embedding.
        metric (str): The distance metric to use: 'euclidean', 'cosine', or 'dot_product'.
        index (faiss.IndexFlat): A FAISS flat index initialized based on the specified metric.
    """

    def __init__(self, dim, metric='euclidean'):
        """
        Initializes the FaissBruteForce index.
        
        Parameters:
            dim (int): The dimensionality of the embeddings.
            metric (str): Distance metric to use. Options are 'euclidean', 'cosine', or 'dot_product'.
        """
        self.dim = int(dim)
        self.metadata = []
        self.metric = metric.lower().strip()

        if self.metric == 'euclidean':
            # Squared L2 distance
            self.index = faiss.IndexFlatL2(self.dim)
        elif self.metric in ('cosine', 'dot_product'):
            # Both cosine similarity and dot product are implemented with the same FAISS primitive
            # For cosine similarity we store normalized vectors and use inner product.
            # For dot product we store raw vectors and use inner product.
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            raise ValueError(
                f"Unsupported metric '{metric}'. Choose from 'euclidean', 'cosine', or 'dot_product'."
            )

    def add_embeddings(self, embeddings, metadata):
        """
        Adds new embeddings and their associated metadata to the index.
        
        Parameters:
            embeddings (list or np.ndarray): A list of embeddings, where each embedding is an array-like
                of length `dim`.
            metadata (list): A list of metadata corresponding to each embedding.
        
        Raises:
            ValueError: If an embedding does not match the specified dimensionality.
            ValueError: If the number of embeddings and metadata entries do not match.
        """
        if embeddings is None:
            raise ValueError("embeddings cannot be None.")
        if metadata is None:
            raise ValueError("metadata cannot be None.")

        # Convert to (n, dim) float32 array, required by FAISS
        emb = np.asarray(embeddings, dtype=np.float32)
        # If a single embedding is given (1D), reshape it to (1, dim).
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        # Verify that each embedding has the correct dimensionality
        if emb.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimensionality mismatch: expected dim={self.dim}, got {emb.shape[1]}."
            )

        # Ensure the number of metadata entries matches the number of embeddings
        if len(metadata) != emb.shape[0]:
            raise ValueError(
                f"Number of embeddings ({emb.shape[0]}) and metadata entries ({len(metadata)}) do not match."
            )

        # For cosine similarity: normalize to unit length and use inner product.
        if self.metric == 'cosine':
            # Compute the L2 norm of each vector (keepdims to allow broadcasting)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            # Identify non-zero vectors to avoid division by zero
            nonzero = norms.squeeze() > 0
            emb_normed = emb.copy()
            # Only normalize vectors that have non-zero norm; zero vectors remain as zero.
            emb_normed[nonzero] = emb[nonzero] / norms[nonzero]
            emb = emb_normed

        # Add the (possibly normalized) embeddings to the FAISS index
        self.index.add(emb)
        # Append the corresponding metadata to our metadata list
        self.metadata.extend(list(metadata))


    def get_metadata(self, idx):
        """
        Retrieves the metadata associated with a particular embedding index.
        
        Parameters:
            idx (int): The index of the embedding.
        
        Returns:
            The metadata associated with the embedding.
        
        Raises:
            IndexError: If the index is out of range.
        """
        if idx < 0 or idx >= len(self.metadata):
            raise IndexError(f"Index {idx} is out of range for metadata of length {len(self.metadata)}.")
        return self.metadata[idx]

    def save(self, filepath):
        """
        Saves the current FaissBruteForce instance to a file.
        
        Parameters:
            filepath (str): The path to the file where the instance should be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Loads a FaissBruteForce instance from a file.
        
        Parameters:
            filepath (str): The path to the file from which to load the instance.
        
        Returns:
            An instance of FaissBruteForce loaded from the file.
        """
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        return instance


if __name__ == "__main__":
    from ironclad.modules.retrieval.search import FaissSearch
    
    # Choose the metric: 'euclidean', 'cosine', or 'dot_product'
    metric = 'cosine'
    index = FaissBruteForce(dim=4, metric=metric)

    # Create some dummy embeddings and corresponding metadata.
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ]
    identity_metadata = [
        "Alice",
        "Bob",
        "Charlie"
    ]

    # Add the embeddings and metadata to the index
    index.add_embeddings(embeddings, identity_metadata)

    # Define a query vector
    query = [0.1, 0.2, 0.3, 0.4]
    k = 2  # number of nearest neighbors to retrieve

    # Perform the search using the dedicated search wrapper
    searcher = FaissSearch(index, metric=metric)
    distances, indices, meta_results = searcher.search(query, k)
    
    print("Query Vector:", query)
    print("Distances:", distances)
    print("Indices:", indices)
    print("Metadata Results:", meta_results)

    # Save the index to disk.
    filepath = "faiss_bruteforce_index.pkl"
    index.save(filepath)
    print(f"Index saved to {filepath}.")

    # Load the index from disk.
    loaded_index = FaissBruteForce.load(filepath)
    print("Loaded Metadata for index 0:", loaded_index.get_metadata(0))

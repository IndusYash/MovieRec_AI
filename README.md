MADAN MOHAN MALAVIYA UNIVERSITY OF TECHNOLOGY

DEPARTMENT OF COMPUTER SCIENCE AND TECHNOLOGY

COMPUTER NETWORK PROJECT

MEMBERS – YASHVARDHAN OJHA AND SAURABH PATEL

Roll no. - 2023021270 and 2023021253
Year - Third Year

//Due to the large size of the dataset used for training the model, the dataset could not be uploaded on the github, refer to the zip  file for the dataset.
🎬 Adaptive Movie Recommender System

Introduction

The Adaptive Movie Recommender System is a large-scale content intelligence engine built to deliver meaningful film suggestions based on narrative themes, mood, and metadata similarity. Unlike simple recommenders that cluster movies using dominant surface words, this system converts each film into a balanced semantic identity that captures plot, genre, cast, crew, and thematic keywords without distortion. The aim is to interpret films as a composition of emotion, storytelling, and cinematic attributes rather than raw text resemblance.

Current Capabilities

🔎 Similarity-Based Recommendations:
Users can search for a movie they enjoy, and the system retrieves the closest cinematic neighbors using cosine K-Nearest Neighbor search.

🎭 Mood-Driven Retrieval:
Users can describe their viewing mood (for example: intense crime thriller, emotional romance, light family comedy), and the system converts it into an embedding query to fetch relevant cinematic matches.

🏷️ Clean Tag Engineering:
Each film is reduced to a dense tag fingerprint built from:

Title (kept intact, not split into separate words)

Genres, cast, crew, and thematic keywords

Lowercase normalization and duplicate removal before embedding

📊 Cosine K-NN Search:
Similarity retrieval is powered by scikit-learn’s NearestNeighbors with cosine distance metric for deterministic and stable recommendations.

🗄️ Metadata-Rich Schema for Future Web Integration:
Film records include:

imdb_id for cross-database lookups

release_date for timeline filtering

runtime for watch-duration gating
These fields are stored as metadata and not repeated or split into embedding text to preserve vector quality.

Planned Enhancements

🎯 User-Selectable Filter System:
After similarity or mood search, users can choose any filter they want to refine recommendations, such as:

Genre

Director

Crew/Keyword overlap

Runtime limits

Release timeline bins

⚡ Scalability Upgrade Path:
Prepared for migration to vector databases (FAISS, HNSWlib, or Chroma) when expanding to web deployment.

🧬 Metadata-Aware Reranking:
Boost recommendation ranking using genre, cast, and crew overlap scoring without distorting embeddings.

⚙ Tech Stack

Language: Python 3.12

Embedding Model: Sentence-BERT (all-MiniLM-L6-v2)

Similarity Search: scikit-learn (NearestNeighbors, cosine metric, K-NN)

Data Engineering: pandas, NumPy

Model Serving (Future Web UI): Streamlit (frontend interface), Flask/FastAPI (optional backend API expansion)

Metadata IDs stored for Web Filters: IMDb ID, release date, runtime

ML Design Paradigm: Content-based recommendation using dense neural embeddings + cosine K-NN retrieval

Scalability Paradigm Prepared For: Vector database indexing (planned, not yet implemented)

Dataset Handling: Safe list/string processing, lowercase normalization, duplicate suppression before vector encoding

🌍 Vision

Aviation Bay observes airspace; APRE observes cinema. The Adaptive Movie Recommender System aims to become a highly relevant, scalable, and intent-aligned film discovery engine. Users decide what similarity means, while the system ensures retrieval stability, metadata balance, and cinematic relevance.



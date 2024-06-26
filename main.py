from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File
from model import extract_feature_vectors
from model2 import calculate_cosine_similarity
from typing import List
from PIL import Image
import io

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    images = [Image.open(io.BytesIO(await file.read())) for file in files]

    feature_vectors = extract_feature_vectors(images)

    feature_vectors = [vector.tolist() if vector is not None else None for vector in feature_vectors]

    return {"feature_vectors": feature_vectors}

@app.post("/calculate_similarity")
async def get_similarity(vector1: List[float], vector2: List[float]):
    similarity_score = calculate_cosine_similarity(vector1, vector2)
    return {"cosine_similarity": similarity_score}

    
if __name__ == "__main__":
    print("Starting FastAPI server...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("FastAPI server started successfully!")

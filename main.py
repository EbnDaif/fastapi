from fastapi import FastAPI, UploadFile, File
from model import extract_feature_vectors
from verify import calculate_similarity
from model2 import calculate_cosine_similarity
from typing import List
from PIL import Image
import io

app = FastAPI()

'''
@app.post("/verify")
async def verify(feature_vectors: List[List[float]]):
    if len(feature_vectors) != 2:
        return {"error": "Exactly two feature vectors are required"}

    result = calculate_similarity(feature_vectors[0], feature_vectors[1])
    print(11)
    return {"result": "Same person" if result else "Different persons"}
'''
'''
@app.post("/predict")
async def predict(image_urls: List[str]):
    images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
    
    feature_vectors = extract_feature_vectors(images)
    
    feature_vectors = [vector.tolist() if vector is not None else None for vector in feature_vectors]
    
    return {"feature_vectors": feature_vectors}
    '''
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

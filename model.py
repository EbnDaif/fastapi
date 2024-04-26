import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def extract_feature_vectors(images):
    vectors = []
    for img in images:
        boxes, probs = mtcnn.detect(img)

        if boxes is None:
            vectors.append(None)
            continue

        img_array = np.array(img)

        faces = [img_array[int(box[1]):int(box[3]), int(box[0]):int(box[2])] for box in boxes]
        embeddings = [resnet(torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()).detach().numpy() for face in faces]

        vectors.extend(embeddings)
    return vectors

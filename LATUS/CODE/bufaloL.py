from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import cv2
import os
import numpy as np


def calc_cost(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

app = FaceAnalysis(name="Buffalo_l",  providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# # Đọc ảnh cần phân tích
# img1 = cv2.imread(r"C:\Users\ATUS\Desktop\code ngu thi chiu\img\negav1.jpg")
# face1 = app.get(img1)[0]

# img2 = cv2.imread(r"C:\Users\ATUS\Desktop\code ngu thi chiu\img\negav.jpg")
# face2 = app.get(img2)[0]


# def show_cropped_faces(img, face):
#     # Lấy bounding box
#     bbox = face.bbox.astype(int)  # Chuyển đổi sang kiểu int
#     x1, y1, x2, y2 = bbox

#     # Cắt khuôn mặt từ ảnh gốc
#     cropped_face = img[y1:y2, x1:x2]

#     # Hiển thị khuôn mặt đã cắt bằng matplotlib
#     plt.imshow(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
#     plt.title(f'Cropped Face')
#     plt.axis('off')
#     plt.show()

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Hiển thị các khuôn mặt đã cắt từ ảnh đầu tiên
# show_cropped_faces(img1, face1)

# # Hiển thị các khuôn mặt đã cắt từ ảnh thứ hai
# show_cropped_faces(img2, face2)

# threshold = 0.5
# if calc_cost(face1.embedding, face2.embedding) >= threshold:
#     print("Hai ảnh là cùng một người")
# else:
#     print("Hai ảnh là khác người")



def toembeddings(img):
    return app.get(img)


def init_embeddings(pathdata, testsize):
    # Đường dẫn tới bộ dữ liệu LFW
    # lfw_path = r"C:\Users\ATUS\Desktop\LUONG ANH TU\Database\lfw"
    # Lưu embeddings của tất cả các khuôn mặt
    embeddings = []
    labels = []

    # Lặp qua từng thư mục trong bộ dữ liệu LFW (mỗi thư mục là một người)
    for person_name in os.listdir(pathdata):
        if (testsize == 0): break
        person_path = os.path.join(pathdata, person_name)
        if os.path.isdir(person_path):
            # Lặp qua từng ảnh của người đó
            for img_name in os.listdir(person_path):
                if (testsize == 0): break
                img_path = os.path.join(person_path, img_name)
                # Đọc ảnh và lấy embedding từ mô hình
                img = cv2.imread(img_path)
                faces = toembeddings(img)
                if len(faces) > 0:
                    face_embedding = faces[0].embedding
                    print(f"{testsize} {person_name}")
                    embeddings.append(face_embedding)
                    labels.append(person_name)
                    testsize-=1
    return embeddings, labels


embeddings, labels = init_embeddings(r"C:\Users\ATUS\Desktop\LUONG ANH TU\Database\lfw", 200)


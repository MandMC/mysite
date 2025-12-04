import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
# 수학 함수와 미분
def f(x):
    return -2*x**3 - 6*x**2 + 18*x
def f_derivative(x):
    return -6*x**2 - 12*x + 18
# 커스텀 활성화 함수 (미분식 기반)
@tf.function
def custom_activation(x):
    # f'(x) = -6x² - 12x + 18을 활성화 함수로 사용
    return -6*x**2 - 12*x + 18
# 커스텀 학습률 조정 (미분 활용)
class CustomOptimizer(tf.keras.optimizers.Adam):
    def _resource_apply_dense(self, grad, var, apply_state=None):
        # 미분값으로 학습률 조정
        lr = self.learning_rate
        x_val = tf.reduce_mean(var)
        # f'(x)를 학습률 스케일링에 활용
        derivative_scale = tf.abs(-6*x_val**2 - 12*x_val + 18) / 20
        adjusted_lr = lr * (1 + derivative_scale)

        var.assign_sub(adjusted_lr * grad)
print("=== 수학 풀이 ===")
print("f(x) = -2x³ - 6x² + 18x")
print("f'(x) = -6x² - 12x + 18")
print("극대: x=1, f(1)=10 | 극소: x=-3, f(-3)=-54")
print("증가: -3<x<1 | 감소: x<-3, x>1\n")
# 이미지 로드
def load_food_images():
    foods = {
        '피자': 'https://images.unsplash.com/photo-1513104890138-7c749659a591?w=400',
        '치킨': 'https://images.unsplash.com/photo-1626082927389-6cd097cdc6ec?w=400',
        '햄버거': 'https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400',
        '파스타': 'https://images.unsplash.com/photo-1621996346565-e3dbc646d9a9?w=400',
        '샐러드': 'https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400'
    }
    images = {}
    for name, url in foods.items():
        try:
            urllib.request.urlretrieve(url, f'{name}.jpg')
            img = Image.open(f'{name}.jpg').resize((224, 224))
            images[name] = np.array(img)
        except:
            images[name] = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    return images
food_images = load_food_images()
foods = list(food_images.keys())
X_train = np.array([food_images[f] for f in foods * 10]) / 255.0
y_train = np.array([i for i in range(5) for _ in range(10)])
# 딥러닝 모델 (미분 활용)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), input_shape=(224,224,3)),
    tf.keras.layers.Activation(custom_activation),   # ← 미분식을 활성화 함수로
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Activation(custom_activation),   # ← 미분식을 활성화 함수로
    tf.keras.layers.Dense(5, activation='softmax')
])
# 미분 기반 커스텀 옵티마이저로 학습
optimizer = CustomOptimizer(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=5, verbose=1)
# 추천
def recommend_food(hunger_level):
    # 딥러닝 예측 (미분이 학습에 사용됨)
    test_img = food_images[foods[0]].reshape(1,224,224,3) / 255.0
    predictions = model.predict(test_img, verbose=0)[0]

    # f'(x) 기반 점수 조정
    scores = []
    for i, pred in enumerate(predictions):
        x = (hunger_level / 10) * 4 - 3 + i * 0.3
        derivative_val = f_derivative(x)
        adjusted = pred * (1 + derivative_val / 20)
        scores.append(adjusted)

    best_idx = np.argmax(scores)
    best_food = foods[best_idx]

    print(f"\n배고픔: {hunger_level}")
    print(f"추천: {best_food} (미분 적용 점수: {scores[best_idx]:.2f})")

    plt.imshow(food_images[best_food])
    plt.title(f'{best_food}\n점수: {scores[best_idx]:.2f}', fontsize=16)
    plt.axis('off')
    plt.show()
recommend_food(hunger_level=6)
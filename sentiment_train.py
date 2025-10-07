import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from underthesea import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ================================================
# 1️⃣ DỮ LIỆU MẪU BAN ĐẦU
# ================================================
data = {
    "text": [
        "Hôm nay tôi rất vui",
        "Trời đẹp quá, tôi cảm thấy hạnh phúc",
        "Tôi ghét phải chờ đợi",
        "Thật tệ, tôi mệt và buồn",
        "Thành công rồi! Tuyệt vời quá",
        "Tôi thấy chán và thất vọng",
        "Cảm ơn bạn, tôi rất hài lòng",
        "Dịch vụ quá tệ, không đáng tiền",
        "Cái bàn này màu xanh",
        "Tôi đang ngồi học",
        "Chúng sinh đau buồn",
        "Cả lớp cùng cười vui vẻ"
    ],
    "label": [
        "positive", "positive", "negative", "negative",
        "positive", "negative", "positive", "negative",
        "neutral", "neutral", "negative", "positive"
    ]
}

df = pd.DataFrame(data)
print(f"📘 Dữ liệu ban đầu: {len(df)} mẫu\n")

# ================================================
# 2️⃣ TIỀN XỬ LÝ NGÔN NGỮ
# ================================================
print("🔧 Đang tiền xử lý văn bản...")

# Tách từ tiếng Việt
df["text"] = df["text"].apply(lambda x: word_tokenize(x, format="text"))

# Tách tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42
)

# Vector hóa văn bản
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✅ Số đặc trưng (features): {len(vectorizer.get_feature_names_out())}")

# ================================================
# 3️⃣ HUẤN LUYỆN MÔ HÌNH
# ================================================
print("\n🚀 Bắt đầu huấn luyện mô hình Logistic Regression...")
model = LogisticRegression(max_iter=200, solver='lbfgs')
model.fit(X_train_vec, y_train)

print("✅ Huấn luyện hoàn tất!")

# ================================================
# 4️⃣ ĐÁNH GIÁ HIỆU SUẤT
# ================================================
print("\n📊 Đánh giá mô hình:")

y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print(f"🎯 Accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred, digits=3))

# Hiển thị confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative", "neutral"])
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=["positive", "negative", "neutral"],
    yticklabels=["positive", "negative", "neutral"]
)
plt.title("Confusion Matrix - Mô hình phân tích cảm xúc (3 lớp)")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.show()

# ================================================
# 5️⃣ LƯU MÔ HÌNH & VECTORIZER
# ================================================
MODEL_FILE = "sentiment_model.pkl"

with open(MODEL_FILE, "wb") as f:
    pickle.dump((model, vectorizer), f)

print(f"\n💾 Đã lưu mô hình vào: {MODEL_FILE}")
print("🏁 Quá trình huấn luyện hoàn tất thành công!")

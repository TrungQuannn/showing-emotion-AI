import streamlit as st
import pandas as pd
import joblib
import pickle
from underthesea import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# 🔧 Cấu hình file dữ liệu
# -----------------------------
DATA_FILE = "sentiment_data.csv"
MODEL_FILE = "sentiment_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

# -----------------------------
# 📦 Load hoặc tạo mới dữ liệu
# -----------------------------
def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        df = pd.DataFrame({"text": [], "label": []})
        df.to_csv(DATA_FILE, index=False)
    return df

# -----------------------------
# 🧠 Load hoặc tạo mới mô hình
# -----------------------------
def load_model():
    try:
        with open(MODEL_FILE, "rb") as f:
            obj = pickle.load(f)
            if isinstance(obj, tuple) and len(obj) == 2:
                model, vectorizer = obj
            else:
                model = joblib.load(MODEL_FILE)
                vectorizer = joblib.load(VECTORIZER_FILE)
    except:
        df = load_data()
        if len(df) > 0:
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(df["text"])
            y = df["label"]
            model = MultinomialNB()
            model.fit(X, y)
            with open(MODEL_FILE, "wb") as f:
                pickle.dump((model, vectorizer), f)
        else:
            vectorizer = CountVectorizer()
            model = MultinomialNB()
    return model, vectorizer

# -----------------------------
# 🚀 Giao diện Streamlit
# -----------------------------
st.set_page_config(page_title="Sentiment Trainer 🇻🇳", page_icon="💬", layout="centered")
st.title("🇻🇳 AI Phân Loại Cảm Xúc Tiếng Việt 😄😞😐")
st.write("AI sẽ học cảm xúc tiếng Việt từ chính bạn — dạy nó thêm khi nó chưa biết nhé!")

# Load dữ liệu & mô hình
df = load_data()
model, vectorizer = load_model()

# -----------------------------
# ✏️ Nhập câu cần phân tích
# -----------------------------
user_text = st.text_input("Nhập câu hoặc từ tiếng Việt:")

if user_text:
    processed = word_tokenize(user_text, format="text")

    # Kiểm tra xem có từ nào chưa học không
    unknown_words = [w for w in processed.split() if w not in vectorizer.vocabulary_]

    if len(unknown_words) > 0:
        st.warning(f"🤔 Tôi chưa học qua {len(unknown_words)} từ: {', '.join(unknown_words)}.")
        st.info("Bạn có muốn dạy tôi biết cảm xúc của từ/câu này không?")

        label = st.radio("Gán nhãn cảm xúc:", ["positive", "negative", "neutral"], horizontal=True)
        if st.button("💾 Lưu từ mới"):
            new_row = pd.DataFrame([[processed, label]], columns=["text", "label"])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            st.success(f"✅ Đã lưu từ mới: '{user_text}' → {label}.")
            st.balloons()
            st.info("💡 Tôi sẽ biết nghĩa từ này sau khi bạn bấm 'Huấn luyện lại mô hình'.")
    else:
        # Nếu tất cả từ đã học, thì dự đoán cảm xúc
        try:
            X_input = vectorizer.transform([processed])
            pred = model.predict(X_input)[0]
            if pred == "positive":
                emoji = "😊"
            elif pred == "negative":
                emoji = "😞"
            else:
                emoji = "😐"
            st.subheader(f"🔍 Kết quả dự đoán: **{pred.upper()}** {emoji}")
        except Exception:
            st.warning("⚠️ Mô hình chưa đủ dữ liệu. Hãy thêm ví dụ và huấn luyện lại.")

# -----------------------------
# 🔁 Huấn luyện lại mô hình
# -----------------------------
st.write("---")
if st.button("🔁 Huấn luyện lại mô hình"):
    df = load_data()
    if len(df) < 3:
        st.warning("⚠️ Cần ít nhất 3 mẫu để huấn luyện.")
    else:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df["text"])
        y = df["label"]
        model = MultinomialNB()
        model.fit(X, y)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump((model, vectorizer), f)
        joblib.dump(vectorizer, VECTORIZER_FILE)
        st.success("✅ À, tôi biết nghĩa của các từ mới rồi! Cảm ơn bạn đã dạy tôi ❤️")

# -----------------------------
# 📊 Hiển thị dữ liệu hiện có
# -----------------------------
st.write("---")
st.subheader("📂 Dữ liệu huấn luyện hiện tại:")
st.dataframe(df.tail(10))

st.caption("💡 Hãy thử nhập: 'Tôi vui quá' (positive), 'Tôi chán lắm' (negative), hoặc 'Tôi đang học' (neutral).")


# -----------------------------
# 💬 BẢNG THỐNG KÊ CẢM XÚC (EMOTION BOARD) - LOGIC ĐÃ ĐƯỢC TINH CHỈNH
# -----------------------------
st.write("---")
st.subheader("📊 Bảng cảm xúc hiện tại")

# Nếu chưa có dữ liệu
if df.empty or df.shape[0] == 0:
    st.write("😶‍🌫️ Ờm... lịch sử chưa có từ mới.")
else:
    counts = df["label"].value_counts()
    pos = int(counts.get("positive", 0))
    neg = int(counts.get("negative", 0))
    neu = int(counts.get("neutral", 0))
    total = pos + neg + neu

    # Tỉ lệ phần trăm
    pos_pct = pos / total * 100 if total > 0 else 0.0
    neg_pct = neg / total * 100 if total > 0 else 0.0
    neu_pct = neu / total * 100 if total > 0 else 0.0

    # Hiển thị biểu đồ thanh (dùng phần trăm cho trực quan)
    chart_data = pd.Series({"positive": pos_pct, "negative": neg_pct, "neutral": neu_pct})
    st.bar_chart(chart_data)

    # ---------- Quy tắc quyết định thông điệp ----------
    # 1) Ba bằng nhau
    if pos == neg == neu:
        emoji = "🤯"
        message = "Ba cảm xúc cân bằng hoàn hảo — đúng kiểu ‘thiền sư’ rồi đó!"
    else:
        # 2) Hai cái bằng nhau và LỚN HƠN cái còn lại (chỉ hiển thị khi đúng điều kiện này)
        if pos == neg and pos > neu:
            emoji = "⚖️"
            message = "Tích cực = Tiêu cực và đều nhiều hơn trung hòa — trái tim bạn nhiều chiều lắm."
        elif pos == neu and pos > neg:
            emoji = "🙂"
            message = "Tích cực = Trung hòa và đều nhiều hơn tiêu cực — sống cân bằng đó!"
        elif neg == neu and neg > pos:
            emoji = "😔"
            message = "Tiêu cực = Trung hòa và đều nhiều hơn tích cực — hơi buồn đấy."
        else:
            # 3) Một cái LỚN HƠN (dominant)
            if pos > neg and pos > neu:
                emoji = "😄"
                message = "Bạn tích cực ghê đó! Toàn năng lượng tốt."
            elif neg > pos and neg > neu:
                emoji = "😞"
                message = "Bạn hơi ủ rũ đấy, nghỉ ngơi chút nha."
            elif neu > pos and neu > neg:
                emoji = "😐"
                message = "Trung hòa - Cầu Giấy vibes luôn."
            else:
                # 4) Các trường hợp còn lại (ví dụ: hai bằng nhau nhưng KHÔNG lớn hơn cái còn lại
                # — trong logic trên, nếu hai bằng nhau nhưng không lớn hơn, thì cái thứ ba sẽ là lớn nhất và đã bị xử lý.
                emoji = "🤷"
                message = "Cảm xúc hơi lộn xộn — không xác định rõ."
    # ---------- Hiển thị kết quả ----------
    st.markdown(f"**Tỉ lệ:** Positive {pos_pct:.1f}%  •  Negative {neg_pct:.1f}%  •  Neutral {neu_pct:.1f}%")
    st.markdown(f"{emoji}  {message}")

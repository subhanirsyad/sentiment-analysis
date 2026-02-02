from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import SentimentModel
from src.preprocess import clean_text

APP_TITLE = "Analisis Sentimen Ulasan Play Store (SVM + TF-IDF)"
ROOT = Path(__file__).resolve().parent


@st.cache_resource
def get_model() -> SentimentModel:
    return SentimentModel().load()


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="centered")
    st.title(APP_TITLE)
    st.caption("Input teks ulasan (Bahasa Indonesia), lalu dapatkan prediksi sentimen **positif/negatif**.")

    try:
        model = get_model()
    except Exception as e:
        st.error(
            "Gagal memuat model. Biasanya ini karena versi dependency tidak cocok dengan file model `.joblib`."
        )
        st.caption("Solusi cepat: pastikan install dependency sesuai `requirements.txt`, lalu deploy ulang.")
        st.exception(e)
        st.stop()

    tabs = st.tabs(["🔎 Prediksi 1 Teks", "📦 Prediksi Batch (CSV)", "ℹ️ Tentang Model"])

    with tabs[0]:
        st.subheader("Prediksi 1 teks")
        default_text = "Aplikasinya sering error dan lemot."
        text = st.text_area("Teks ulasan", value=default_text, height=120)

        col1, col2 = st.columns([1, 1])
        with col1:
            do_predict = st.button("Prediksi", type="primary", use_container_width=True)
        with col2:
            st.write("")

        if do_predict:
            if not str(text).strip():
                st.warning("Tolong isi teks ulasan dulu.")
            else:
                label, margin = model.predict_one(text)
                cleaned = clean_text(text)

                st.success(f"Hasil: **{label.upper()}**")
                st.write("Teks setelah dibersihkan (sesuai preprocessing saat training):")
                st.code(cleaned)

                if margin == margin:  # not NaN
                    st.write(f"Skor margin (decision_function): `{margin:.4f}`")
                    st.caption(
                        "Catatan: ini **bukan** probabilitas. Nilai absolut yang lebih besar biasanya berarti model lebih yakin."
                    )

    with tabs[1]:
        st.subheader("Prediksi batch dari file CSV")
        st.write("Upload CSV berisi kolom `content` (teks ulasan) atau `text` (teks yang sudah dibersihkan).")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview data:")
            st.dataframe(df.head(20), use_container_width=True)

            # Tentukan kolom input
            col_candidates = [c for c in ["content", "text", "review", "ulasan"] if c in df.columns]
            if not col_candidates:
                st.error("CSV tidak memiliki kolom `content` atau `text`. Silakan sesuaikan kolom input.")
            else:
                input_col = st.selectbox("Pilih kolom teks", options=col_candidates, index=0)
                run_batch = st.button("Jalankan Prediksi Batch", type="primary")

                if run_batch:
                    texts = df[input_col].astype(str).fillna("").tolist()
                    preds = model.predict_many(texts)

                    out = df.copy()
                    out["prediksi_sentimen"] = preds

                    st.success("Selesai! Anda bisa download hasilnya.")
                    st.dataframe(out.head(50), use_container_width=True)

                    buf = io.StringIO()
                    out.to_csv(buf, index=False)
                    st.download_button(
                        "Download hasil (CSV)",
                        data=buf.getvalue().encode("utf-8"),
                        file_name="hasil_prediksi_sentimen.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    with tabs[2]:
        st.subheader("Tentang Model")
        st.markdown(
            """
- **Model:** LinearSVC (SVM linear)  
- **Fitur:** TF-IDF n-gram (1–2)  
- **Label:** `positif` / `negatif` (berdasarkan rating bintang 1–2 vs 4–5)  
- **Sumber data:** ulasan Google Play via `google-play-scraper`  
            """
        )

        st.write("File model:")
        st.code(str(ROOT / "models" / "svm_tfidf.joblib"))

        st.write("Contoh preprocessing (yang dipakai saat training):")
        st.code(clean_text("Fiturnya bagus banget!!! https://contoh.com"))


if __name__ == "__main__":
    main()

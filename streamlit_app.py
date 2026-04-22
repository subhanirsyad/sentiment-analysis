from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.model import SentimentModel
from src.preprocess import clean_text

APP_TITLE = "Analisis Sentimen Ulasan Play Store"
ROOT = Path(__file__).resolve().parent

EXAMPLE_TEXTS = [
    "Aplikasi ini sangat membantu, tampilannya bersih dan cepat diakses.",
    "Fitur barunya bagus, tapi notifikasi masih sering telat.",
    "Sering crash sendiri dan baterai jadi boros, sangat mengecewakan.",
    "Update terbaru malah bikin lemot, lebih baik versi sebelumnya.",
]


@st.cache_resource
def get_model() -> SentimentModel:
    return SentimentModel().load()


def confidence_label(margin: float) -> str:
    val = abs(margin)
    if val >= 1.5:
        return "Sangat yakin"
    elif val >= 0.7:
        return "Cukup yakin"
    return "Prediksi marginal"


def render_hero():
    st.markdown(
        """
        <div style="
            background: linear-gradient(160deg, #1E293B 0%, #0F172A 100%);
            border-top: 3px solid #3B82F6;
            border-radius: 12px;
            padding: 36px 32px 28px 32px;
            margin-bottom: 28px;
        ">
            <div style="display:flex; gap:8px; margin-bottom:12px;">
                <span style="background:#3B82F6; color:#fff; font-size:0.7rem; font-weight:700;
                    padding:3px 10px; border-radius:99px; letter-spacing:1px; text-transform:uppercase;">NLP</span>
                <span style="background:transparent; color:#64748B; font-size:0.7rem; font-weight:600;
                    padding:3px 10px; border-radius:99px; border:1px solid #334155;
                    letter-spacing:1px; text-transform:uppercase;">Bahasa Indonesia</span>
            </div>
            <h1 style="color:#F1F5F9; font-size:1.75rem; font-weight:800; margin:0 0 8px 0; line-height:1.25;">
                Analisis Sentimen<br>Ulasan Google Play Store
            </h1>
            <p style="color:#64748B; font-size:0.88rem; margin:0; line-height:1.7;">
                Klasifikasi otomatis ulasan berbahasa Indonesia ke dalam kategori
                <strong style="color:#4ADE80;">positif</strong> atau <strong style="color:#F87171;">negatif</strong>
                menggunakan machine learning berbasis teks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pendahuluan():
    st.markdown(
        """
        <div style="margin-bottom:28px;">
            <p style="color:#64748B; font-size:0.88rem; line-height:1.9; margin:0;">
                Jutaan pengguna meninggalkan ulasan di Google Play Store setiap harinya — membacanya satu per satu
                tidak praktis. Data ulasan dikumpulkan via <code style="color:#93C5FD;">google-play-scraper</code>
                dan dilabeli berdasarkan rating bintang: ulasan berbintang 1–2 dikategorikan sebagai
                <strong style="color:#F87171;">negatif</strong>, ulasan berbintang 4–5 sebagai
                <strong style="color:#4ADE80;">positif</strong>, sedangkan bintang 3 tidak digunakan karena
                cenderung ambigu. Setiap teks kemudian diubah ke representasi numerik menggunakan
                <strong style="color:#93C5FD;">TF-IDF</strong> dengan kombinasi unigram dan bigram,
                lalu diklasifikasi menggunakan <strong style="color:#93C5FD;">LinearSVC</strong> —
                pendekatan klasik yang ringan dan cepat tanpa membutuhkan deep learning.
                Skor yang ditampilkan pada hasil prediksi adalah <em>decision function margin</em>,
                bukan probabilitas — nilai absolutnya yang lebih besar menandakan model lebih yakin
                terhadap prediksinya.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-bottom:32px;'></div>", unsafe_allow_html=True)


def render_result(label: str, margin: float, original_text: str, cleaned: str):
    is_positive = label.lower() == "positif"
    color_bg = "#052e16" if is_positive else "#2d0a0a"
    color_border = "#16A34A" if is_positive else "#DC2626"
    color_text = "#4ADE80" if is_positive else "#F87171"
    label_display = "Positif" if is_positive else "Negatif"

    clamped = max(-3.0, min(3.0, float(margin)))
    confidence = int((abs(clamped) / 3.0) * 100)
    c_label = confidence_label(margin)

    token_asli = len(original_text.split())
    token_bersih = len(cleaned.split()) if cleaned.strip() else 0

    st.markdown(
        f"""
        <div style="
            background:{color_bg};
            border-left:4px solid {color_border};
            border-radius:10px;
            padding:20px 24px;
            margin-top:20px;
        ">
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px;">
                <div>
                    <div style="color:#94A3B8; font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:1px; margin-bottom:4px;">Hasil Prediksi</div>
                    <div style="color:{color_text}; font-size:1.75rem; font-weight:800;">{label_display}</div>
                    <div style="color:#64748B; font-size:0.8rem; margin-top:2px;">
                        {c_label} &middot; margin
                        <code style="color:#93C5FD; background:transparent;">{margin:.4f}</code>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="color:#94A3B8; font-size:0.72rem; text-transform:uppercase;
                        letter-spacing:1px; margin-bottom:4px;">Tingkat Keyakinan</div>
                    <div style="color:{color_text}; font-size:1.75rem; font-weight:800;">{confidence}%</div>
                    <div style="color:#64748B; font-size:0.8rem; margin-top:2px;">
                        {token_asli} kata &rarr; {token_bersih} token bersih
                    </div>
                </div>
            </div>
            <div style="margin-top:14px; background:#0F172A; border-radius:99px; height:5px; overflow:hidden;">
                <div style="width:{confidence}%; height:5px; background:{color_border}; border-radius:99px;"></div>
            </div>
        </div>
        <div style="margin-top:10px; color:#475569; font-size:0.78rem; padding:0 4px;">
            Teks setelah preprocessing:
            <code style="color:#93C5FD; background:#1E293B; padding:2px 8px; border-radius:4px; margin-left:4px;">
                {cleaned if cleaned.strip() else "(kosong)"}
            </code>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediksi(model: SentimentModel):
    st.markdown(
        """
        <h2 style="color:#F1F5F9; font-size:1.1rem; font-weight:700; margin:0 0 16px 0;">
            Coba Prediksi
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='color:#64748B; font-size:0.78rem; margin-bottom:6px;'>Coba dengan contoh ulasan:</div>",
        unsafe_allow_html=True,
    )
    ex_cols = st.columns(2)
    for i, ex in enumerate(EXAMPLE_TEXTS):
        with ex_cols[i % 2]:
            if st.button(ex[:48] + "…", key=f"ex_{i}", use_container_width=True):
                st.session_state.input_text = ex

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    text = st.text_area(
        "Teks ulasan",
        value=st.session_state.get("input_text", ""),
        height=120,
        placeholder="Tulis ulasan di sini…",
        key="input_text",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        do_predict = st.button("Analisis Sentimen", type="primary", use_container_width=True)
    with c2:
        do_clear = st.button("Reset", use_container_width=True)

    if do_clear:
        st.session_state.input_text = ""
        st.rerun()

    if do_predict:
        if not str(text).strip():
            st.warning("Teks ulasan belum diisi.")
        else:
            label, margin = model.predict_one(text)
            cleaned = clean_text(text)
            render_result(label, margin, text, cleaned)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")

    render_hero()
    render_pendahuluan()

    try:
        model = get_model()
    except Exception as e:
        st.error("Gagal memuat model. Pastikan dependency sudah terinstall sesuai `requirements.txt`.")
        st.exception(e)
        st.stop()

    render_prediksi(model)


if __name__ == "__main__":
    main()

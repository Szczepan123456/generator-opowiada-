import streamlit as st
import openai
import base64
import requests
from qdrant_client import QdrantClient
from uuid import uuid4

st.set_page_config(page_title="📖 Generator Opowieści", layout="centered")
st.title("📖 Generator Opowieści dla Dzieci i Dorosłych")

# --- Sidebar ---
with st.sidebar:
    api_key = st.text_input("🔑 Wprowadź swój OpenAI API Key:", type="password", value=st.secrets["openai"]["api_key"])
    audience = st.radio("Wybierz odbiorcę opowieści:", ["Dziecko", "Dorosły"])

    if audience == "Dziecko":
        categories = [
            "Baśnie i legendy – klasyczne historie z morałem, pełne magii, zwierząt mówiących, fantastycznych postaci.",
            "Przyjaźń i rodzina – opowieści o relacjach, współpracy, wsparciu i miłości.",
            "Przygoda i odkrywanie świata – historie pełne ciekawych miejsc i wyzwań.",
            "Nauka i edukacja – opowieści uczące liczenia, liter, wartości czy wiedzy o przyrodzie.",
            "Fantastyka i magia – czarodzieje, smoki, wróżki, baśniowe światy.",
            "Zwierzęta i natura – historie o zwierzętach, przyrodzie, ekologii.",
            "Rozwiązywanie problemów i wartości – opowieści uczące radzenia sobie z emocjami, uczciwości, odwagi."
        ]
    else:
        categories = [
            "Romans – miłość, związki, relacje międzyludzkie, czasem złożone emocje.",
            "Dramat i psychologia – historie poruszające kwestie egzystencjalne, problemy społeczne, wewnętrzne konflikty bohaterów.",
            "Kryminał i thriller – zagadki, napięcie, zagrożenia, tajemnice.",
            "Fantastyka i science fiction – zaawansowane światy, przyszłość, technologia, filozoficzne pytania.",
            "Horror – strach, groza, napięcie.",
            "Historia i literatura faktu – opowieści osadzone w realnych wydarzeniach, biografie, reportaże.",
            "Komedia i satyra – humor, ironia, krytyka społeczna.",
            "Filozofia i refleksja – opowieści skłaniające do myślenia, metaforyczne, symboliczne."
        ]
    category = st.selectbox("Wybierz kategorię opowieści:", categories)

    if st.button("🔄 Resetuj wszystko"):
        for key in ["title", "summary", "story", "image_url", "step", "topic"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# --- Sprawdzenie klucza API ---
if not api_key:
    st.warning("Wprowadź klucz API, aby rozpocząć.")
    st.stop()

openai.api_key = api_key

# --- Inicjalizacja Qdrant ---
qdrant_client = QdrantClient(
    url=st.secrets["qdrant"]["url"],
    api_key=st.secrets["qdrant"]["api_key"]
)

# Utworzenie kolekcji jeśli nie istnieje
try:
    qdrant_client.get_collection("stories")
except Exception:
    qdrant_client.recreate_collection(
        collection_name="stories",
        vectors_config={"size": 1536, "distance": "Cosine"}
    )

# --- Stan sesji ---
default_state = {
    "title": "",
    "summary": "",
    "story": "",
    "image_url": "",
    "step": "start",
    "topic": ""
}

for key, val in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- Funkcje ---

def generate_title_and_summary_from_topic(topic_text):
    prompt = (
        f"Napisz tytuł (krótki, chwytliwy) i jednozdaniowe streszczenie opowieści na temat: {topic_text}\n"
        "Format:\nTytuł: ...\nStreszczenie: ..."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def generate_story_full(topic, audience, category):
    prompt_story = (
        f'Napisz opowieść dla {"dziecka" if audience == "Dziecko" else "dorosłego"} na temat: "{topic}". '
        f"Kategoria opowieści: {category}. "
        "Opowieść powinna mieć wyraźny początek, środek i zakończenie, w około 3 akapitach."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_story}],
        temperature=0.8,
        max_tokens=700
    )
    return response.choices[0].message.content.strip()

def parse_title_and_summary(text):
    title = ""
    summary = ""
    for line in text.splitlines():
        if line.lower().startswith("tytuł:"):
            title = line.split(":",1)[1].strip()
        elif line.lower().startswith("streszczenie:"):
            summary = line.split(":",1)[1].strip()
    return title, summary

def generate_image(prompt_img):
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt_img,
        n=1,
        size="1024x1024"
    )
    return response['data'][0]['url']

def estimate_cost_story():
    return 700 * 0.03 / 1000

def estimate_cost_image():
    return 0.02

def download_button(text, filename, label):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_image(url, filename="obraz.png"):
    response = requests.get(url)
    b64 = base64.b64encode(response.content).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Pobierz ilustrację</a>'
    st.markdown(href, unsafe_allow_html=True)

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response["data"][0]["embedding"]

def add_story_to_qdrant(story_id, title, summary, audience, category):
    vector = get_embedding(f"{title} {summary}")
    qdrant_client.upsert(
        collection_name="stories",
        points=[
            {
                "id": story_id,
                "vector": vector,
                "payload": {
                    "type": "story",
                    "title": title,
                    "summary": summary,
                    "audience": audience,
                    "category": category,
                },
            }
        ],
    )
    st.success("✅ Opowieść została zapisana w Qdrant!")

def add_image_to_qdrant(image_id, story_id, image_url, prompt_img):
    vector = get_embedding(prompt_img)
    qdrant_client.upsert(
        collection_name="stories",
        points=[
            {
                "id": image_id,
                "vector": vector,
                "payload": {
                    "type": "image",
                    "story_id": story_id,
                    "image_url": image_url,
                    "prompt": prompt_img
                },
            }
        ],
    )
    st.success("✅ Ilustracja została zapisana w Qdrant!")

# --- Flow aplikacji ---
if st.session_state.step == "start":
    topic = st.text_input("✍️ Wprowadź temat opowieści:")

    if st.button("🎉 Generuj Tytuł i Streszczenie"):
        if topic.strip():
            with st.spinner("Generowanie tytułu i streszczenia..."):
                ts_text = generate_title_and_summary_from_topic(topic)
                title, summary = parse_title_and_summary(ts_text)
                st.session_state.title = title
                st.session_state.summary = summary
                st.session_state.topic = topic
                st.session_state.step = "title_confirm"
            st.experimental_rerun()

elif st.session_state.step == "title_confirm":
    st.subheader("Proponowany tytuł i streszczenie:")
    st.markdown(f"**Tytuł:** {st.session_state.title}")
    st.markdown(f"**Streszczenie:** {st.session_state.summary}")
    download_button(f"Tytuł: {st.session_state.title}\nStreszczenie: {st.session_state.summary}", "tytul_i_streszczenie.txt", "📥 Pobierz")

    cost_story = estimate_cost_story()
    st.info(f"Szacunkowy koszt wygenerowania opowieści: ${cost_story:.4f}")

    if st.button("OK, akceptuję tytuł"):
        with st.spinner("Generowanie opowieści..."):
            story = generate_story_full(st.session_state.topic, audience, category)
            st.session_state.story = story

            # Zapis opowieści w Qdrant
            story_id = str(uuid4())
            add_story_to_qdrant(story_id, st.session_state.title, st.session_state.summary, audience, category)
            st.session_state.story_id = story_id
            st.session_state.step = "story_generated"
        st.experimental_rerun()

elif st.session_state.step == "story_generated":
    st.subheader("Opowieść:")
    st.markdown(st.session_state.story)
    download_button(st.session_state.story, "opowiesc.txt", "📥 Pobierz opowieść")

    if st.button("🎨 Generuj ilustrację"):
        with st.spinner("Generowanie ilustracji..."):
            prompt_img = f"Ilustracja do opowieści: {st.session_state.title}"
            url = generate_image(prompt_img)
            st.session_state.image_url = url

            # Zapis ilustracji w Qdrant (powiązanie z opowieścią)
            image_id = str(uuid4())
            add_image_to_qdrant(image_id, st.session_state.story_id, url, prompt_img)

        st.experimental_rerun()

    if st.session_state.image_url:
        st.image(st.session_state.image_url, caption="Ilustracja do opowieści")
        download_image(st.session_state.image_url)

    cost_image = estimate_cost_image()
    st.info(f"Szacunkowy koszt wygenerowania ilustracji: ${cost_image:.4f}")

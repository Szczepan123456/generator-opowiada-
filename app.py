import streamlit as st
import base64
import requests
from uuid import uuid4
from qdrant_client import QdrantClient
from openai import OpenAI

# Konfiguracja strony
st.set_page_config(page_title="📖 Generator Opowieści", layout="centered")
st.title("📖 Generator Opowieści dla Dzieci i Dorosłych")

# Sidebar – klucz API i ustawienia
with st.sidebar:
    st.markdown("### 🔑 Wprowadź swój OpenAI API Key")
    api_key = st.text_input("Klucz API:", type="password")

    audience = st.radio("Wybierz odbiorcę opowieści:", ["Dziecko", "Dorosły"])

    categories = (
        ["Baśnie i legendy...", "Przyjaźń i rodzina...", "Przygoda i odkrywanie...", "Nauka i edukacja...", "Fantastyka i magia...", "Zwierzęta i natura...", "Rozwiązywanie problemów..."]
        if audience == "Dziecko"
        else ["Romans...", "Dramat i psychologia...", "Kryminał i thriller...", "Fantastyka i sci-fi...", "Horror...", "Historia i fakt...", "Komedia i satyra...", "Filozofia i refleksja..."]
    )
    category = st.selectbox("Wybierz kategorię opowieści:", categories)

    if st.button("🔄 Resetuj wszystko"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

# Wymagany klucz API
if not api_key:
    st.warning("Wprowadź klucz API, aby rozpocząć.")
    st.stop()

# Inicjalizacja klientów
client = OpenAI(api_key=api_key)

# Qdrant: dane dostępowe trzymasz TYLKO w Streamlit Secrets (nie są nigdzie automatycznie używane)
qdrant_client = QdrantClient(
    url=st.secrets["qdrant"]["url"],
    api_key=st.secrets["qdrant"]["api_key"]
)

# Tworzenie kolekcji w Qdrant (jeśli nie istnieje)
if "stories" not in [col.name for col in qdrant_client.get_collections().collections]:
    qdrant_client.recreate_collection(
        collection_name="stories",
        vectors_config={"size": 1536, "distance": "Cosine"}
    )

# Stan sesji
for key in ["title", "summary", "story", "image_url", "step", "topic", "story_id"]:
    st.session_state.setdefault(key, "")
if not st.session_state.step:
    st.session_state.step = "start"

# Funkcje generujące
def generate_title_and_summary_from_topic(topic):
    prompt = f"Napisz tytuł i streszczenie na temat: {topic}\nFormat:\nTytuł: ...\nStreszczenie: ..."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def parse_title_and_summary(text):
    title, summary = "", ""
    for line in text.splitlines():
        if line.lower().startswith("tytuł:"):
            title = line.split(":",1)[1].strip()
        elif line.lower().startswith("streszczenie:"):
            summary = line.split(":",1)[1].strip()
    return title, summary

def generate_story_full(topic, audience, category):
    prompt = f"Napisz opowieść dla {'dziecka' if audience == 'Dziecko' else 'dorosłego'} na temat '{topic}', kategoria: {category}."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=700
    )
    return response.choices[0].message.content.strip()

def generate_image(prompt_img):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt_img,
        n=1,
        size="1024x1024"
    )
    return response.data[0].url

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def add_story_to_qdrant(story_id, title, summary, audience, category):
    vector = get_embedding(f"{title} {summary}")
    qdrant_client.upsert(
        collection_name="stories",
        points=[{
            "id": story_id,
            "vector": vector,
            "payload": {
                "type": "story",
                "title": title,
                "summary": summary,
                "audience": audience,
                "category": category
            }
        }]
    )

def add_image_to_qdrant(image_id, story_id, image_url, prompt_img):
    vector = get_embedding(prompt_img)
    qdrant_client.upsert(
        collection_name="stories",
        points=[{
            "id": image_id,
            "vector": vector,
            "payload": {
                "type": "image",
                "story_id": story_id,
                "image_url": image_url,
                "prompt": prompt_img
            }
        }]
    )

def download_button(text, filename, label):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_image(url, filename="obraz.png"):
    response = requests.get(url)
    b64 = base64.b64encode(response.content).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📅 Pobierz ilustrację</a>'
    st.markdown(href, unsafe_allow_html=True)

# Flow aplikacji
if st.session_state.step == "start":
    topic = st.text_input("✍️ Wprowadź temat opowieści:")
    if st.button("🎉 Generuj Tytuł i Streszczenie") and topic.strip():
        with st.spinner("Generowanie..."):
            output = generate_title_and_summary_from_topic(topic)
            title, summary = parse_title_and_summary(output)
            st.session_state.update({"title": title, "summary": summary, "topic": topic, "step": "title_confirm"})
        st.rerun()

elif st.session_state.step == "title_confirm":
    st.subheader("Proponowany tytuł i streszczenie:")
    st.markdown(f"**Tytuł:** {st.session_state.title}")
    st.markdown(f"**Streszczenie:** {st.session_state.summary}")
    download_button(f"Tytuł: {st.session_state.title}\nStreszczenie: {st.session_state.summary}", "tytul.txt", "📅 Pobierz")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("OK, akceptuj tytuł"):
            with st.spinner("Generowanie opowieści..."):
                story = generate_story_full(st.session_state.topic, audience, category)
                story_id = str(uuid4())
                add_story_to_qdrant(story_id, st.session_state.title, st.session_state.summary, audience, category)
                st.session_state.update({"story": story, "step": "story_generated", "story_id": story_id})
            st.rerun()
    with col2:
        if st.button("Nie akceptuję tytułu, proszę o nowy"):
            with st.spinner("Generowanie nowego tytułu i streszczenia..."):
                output = generate_title_and_summary_from_topic(st.session_state.topic)
                title, summary = parse_title_and_summary(output)
                st.session_state.update({"title": title, "summary": summary})
            st.rerun()

elif st.session_state.step == "story_generated":
    st.subheader("Opowieść:")
    st.markdown(st.session_state.story)
    download_button(st.session_state.story, "opowiesc.txt", "📅 Pobierz opowieść")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎨 Generuj ilustrację"):
            if not st.session_state.get("story_id"):
                st.warning("Najpierw wygeneruj opowieść.")
            else:
                with st.spinner("Generowanie ilustracji..."):
                    prompt_img = f"Ilustracja w stylu bajkowym do opowieści pt. '{st.session_state.title}'. Krótkie streszczenie: {st.session_state.summary}"
                    url = generate_image(prompt_img)
                    image_id = str(uuid4())
                    add_image_to_qdrant(image_id, st.session_state.story_id, url, prompt_img)
                    st.session_state.image_url = url
                st.rerun()
    with col2:
        if st.session_state.image_url and st.button("Nie akceptuję ilustracji, proszę o nową"):
            with st.spinner("Generowanie nowej ilustracji..."):
                prompt_img = f"Ilustracja w stylu bajkowym do opowieści pt. '{st.session_state.title}'. Krótkie streszczenie: {st.session_state.summary}"
                url = generate_image(prompt_img)
                image_id = str(uuid4())
                add_image_to_qdrant(image_id, st.session_state.story_id, url, prompt_img)
                st.session_state.image_url = url
            st.rerun()

    if st.session_state.image_url:
        st.image(st.session_state.image_url, caption="Ilustracja do opowieści")
        download_image(st.session_state.image_url)

import streamlit as st
import pickle as pk
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(
    page_title="COSINE.CINEMA | Movie Discovery",
    page_icon="🎬",
    layout="wide"
)
@st.cache_data(ttl=86400)
def get_movie_details(title):
    api_key = st.secrets.get("TMDB_API_KEY", "")

    if not api_key or not title:
        return None

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": api_key,
                "query": title
            },
            timeout=10
        )
        response.raise_for_status()

        results = response.json().get("results", [])

        if not results:
            return None

        # Use the first result with the most useful details.
        result = next(
            (m for m in results if m.get("poster_path")),
            results[0]
        )

        poster_path = result.get("poster_path")

        return {
            "poster": (
                f"https://image.tmdb.org/t/p/w500{poster_path}"
                if poster_path else None
            ),
            "overview": result.get("overview") or "No description available.",
            "year": (result.get("release_date") or "")[:4],
            "rating": result.get("vote_average")
        }

    except (requests.RequestException, ValueError):
        return None


st.markdown("""
<style>
/* Import a bold, modern font */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], button, input {
    font-family: 'Montserrat', sans-serif !important;
}

/* Page background */
.stApp {
    background: linear-gradient(
        135deg,
        #10060A 0%,
        #3B0A16 38%,
        #101A3A 75%,
        #070B18 100%
    );
    color: #F5F5FF;
}

/* Main title */
h1 {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #FF4545, #F4F4FF, #5B8CFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Section headings */
h2, h3 {
    color: #F4F4FF !important;
    font-weight: 700 !important;
}

/* General text */
p, label, .stMarkdown, .stText {
    color: #E2E7F5;
    font-weight: 500;
}

/* Recommendation button */
.stButton > button {
    background: linear-gradient(100deg, #D71935 0%, #7D244B 48%, #245BEA 100%) !important;
    color: #FFFFFF !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.4rem !important;
    transition: all 0.25s ease !important;
}

/* Button hover */
.stButton > button:hover {
    background: linear-gradient(100deg, #FF3048, #356FFF) !important;
    box-shadow: 0 0 20px rgba(53, 111, 255, 0.3);
    transform: translateY(-2px);
}

/* Movie cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(
        145deg,
        rgba(75, 15, 30, 0.82),
        rgba(15, 27, 60, 0.92)
    );
    border: 1px solid rgba(120, 150, 255, 0.25);
    border-radius: 14px;
}

/* Movie selector */
div[data-baseweb="select"] > div {
    background-color: #171525;
    border: 1px solid #394B83;
    border-radius: 9px;
}

/* Selected movie text */
div[data-baseweb="select"] * {
    font-weight: 600;
}

/* Expandable descriptions */
div[data-testid="stExpander"] {
    background: rgba(14, 18, 38, 0.75);
    border: 1px solid rgba(100, 130, 220, 0.3);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
movie = pk.load(open('movie.pkl', 'rb'))
tfidf_matrix = pk.load(open('tfidf_matrix.pkl', 'rb'))

st.markdown("# COSINE<span style='color:#e04462'>.</span>CINEMA", unsafe_allow_html=True)
st.markdown(
    '<p class="section-caption">'
    'Discover your next favourite film.'
    '</p>',
    unsafe_allow_html=True
)
num_recommendations = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=5,
    step=1
)
selected_movie = st.selectbox('Select movies', movie['names'].values)

def recommend(sel_movie, n=5):
    sel_movie = sel_movie.lower()
    idx = movie[movie['names'] == sel_movie].index[0]

    sims = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    dist = sorted(
        enumerate(sims),
        reverse=True,
        key=lambda x: x[1]
    )

    return [
        movie.iloc[i].names
        for i, score in dist
        if i != idx
    ][:n]


if st.button("Show Recommendations"):
    movie_names = recommend(selected_movie, num_recommendations)

    st.markdown("### Recommended for you")
    st.markdown(
        '<p class="section-caption">'
        'Films selected based on content similarity.'
        '</p>',
        unsafe_allow_html=True
    )

    # Display five movies per row
    for start in range(0, len(movie_names), 5):
        cols = st.columns(5)

        for col, name in zip(cols, movie_names[start:start + 5]):
            with col:
                details = get_movie_details(name)

                if details:
                    poster = details.get("poster")

                    if poster and poster.startswith("https://"):
                        st.image(
                            poster,
                            use_container_width=True
                        )
                    else:
                        st.caption("Poster unavailable")

                    st.markdown(f"**{name}**")

                    if details.get("year"):
                        st.caption(details["year"])

                    rating = details.get("rating")
                    if rating is not None and rating > 0:
                        st.caption(f"TMDB rating: {rating:.1f}/10")

                    with st.expander("Movie description"):
                        st.write(
                            details.get("overview")
                            or "No description available."
                        )

                else:
                    st.markdown(f"**{name}**")
                    st.caption("Movie details unavailable")

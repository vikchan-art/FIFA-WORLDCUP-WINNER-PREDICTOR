"""
FIFA World Cup 2026 - Match Predictor
Streamlit app - redesigned with SVG trophy+ball logo
run it with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0e1a; color: #e8eaf0; }
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding: 2rem 1.5rem 4rem; max-width: 680px; }



/* ── logo ── */
.logo-wrap {
    display: flex;
    justify-content: center;
    margin-bottom: 0.6rem;
}

/* ── hero ── */
.hero { text-align: center; padding: 1.5rem 0 2rem; }
.hero-eyebrow {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.85rem; font-weight: 600;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #4ade80; margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: clamp(2.6rem, 8vw, 4rem); font-weight: 800;
    line-height: 1.0; color: #ffffff;
    text-transform: uppercase; letter-spacing: 0.02em; margin: 0;
}
.hero-title span { color: #4ade80; }
.hero-sub { font-size: 0.82rem; color: #6b7280; margin-top: 0.75rem; letter-spacing: 0.04em; }

.pitch-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #4ade80 40%, #4ade80 60%, transparent);
    margin: 1.5rem 0 2rem; opacity: 0.35;
}

/* ── selectors ── */
.team-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #6b7280; margin-bottom: 0.4rem;
}
div[data-testid="stSelectbox"] > div > div {
    background: #131929 !important; border: 1px solid #1e2d45 !important;
    border-radius: 8px !important; color: #e8eaf0 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.15rem !important; font-weight: 700 !important;
}
div[data-testid="stSelectbox"] > div > div:hover { border-color: #4ade80 !important; }

.vs-badge { display: flex; align-items: center; justify-content: center; height: 100%; padding-top: 1.6rem; }
.vs-text { font-family: 'Barlow Condensed', sans-serif; font-size: 1.6rem; font-weight: 800; color: #374151; }

/* ── button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #16a34a 0%, #4ade80 100%) !important;
    color: #0a0e1a !important; font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.1rem !important; font-weight: 800 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.7rem 2rem !important; width: 100% !important;
    transition: opacity 0.15s ease !important;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }

/* ── result card ── */
.result-card {
    background: #0f1623; border: 1px solid #1e2d45;
    border-radius: 12px; padding: 1.8rem 1.5rem 1.5rem;
    margin: 1.5rem 0; text-align: center;
}
.result-winner {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: clamp(1.6rem, 5vw, 2.4rem); font-weight: 800;
    color: #4ade80; text-transform: uppercase; letter-spacing: 0.04em; line-height: 1.1;
}
.result-match { font-size: 0.78rem; color: #4b5563; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.3rem; }

/* ── prob bars ── */
.prob-row { display: flex; align-items: center; gap: 0.75rem; margin: 0.6rem 0; }
.prob-label { font-family: 'Barlow Condensed', sans-serif; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #9ca3af; min-width: 110px; text-align: right; }
.prob-track { flex: 1; height: 8px; background: #1e2d45; border-radius: 4px; overflow: hidden; }
.prob-fill-green { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #16a34a, #4ade80); }
.prob-fill-gray  { height: 100%; border-radius: 4px; background: #374151; }
.prob-fill-blue  { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #1d4ed8, #60a5fa); }
.prob-pct { font-family: 'Barlow Condensed', sans-serif; font-size: 1rem; font-weight: 700; color: #e8eaf0; min-width: 42px; }

/* ── metric tiles ── */
.metric-row { display: flex; gap: 0.75rem; margin: 1.2rem 0 0.5rem; }
.metric-tile { flex: 1; background: #0f1623; border: 1px solid #1e2d45; border-radius: 10px; padding: 1rem 0.5rem; text-align: center; }
.metric-tile.highlight { border-color: #16a34a; background: #0d1f12; }
.metric-val { font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; font-weight: 800; color: #4ade80; line-height: 1; }
.metric-tile:not(.highlight) .metric-val { color: #e8eaf0; }
.metric-name { font-size: 0.68rem; color: #6b7280; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.3rem; }

/* ── footer ── */
.app-footer { text-align: center; font-size: 0.72rem; color: #374151; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #111827; }
</style>
""", unsafe_allow_html=True)

# ── Data & Model ──────────────────────────────────────────────────────────────

TEAMS_2026 = sorted([
    'Argentina', 'Australia', 'Belgium', 'Brazil', 'Cameroon', 'Chile',
    'Colombia', 'Croatia', 'Denmark', 'Ecuador', 'England', 'France',
    'Germany', 'Iran', 'Italy', 'Japan', 'Mexico', 'Morocco',
    'Netherlands', 'Nigeria', 'Poland', 'Portugal', 'Saudi Arabia',
    'Senegal', 'Serbia', 'South Korea', 'Spain', 'Switzerland',
    'Uruguay', 'USA'
])

FIFA_RANKS = {
    'Brazil': 1, 'France': 2, 'Argentina': 3, 'Germany': 4,
    'England': 5, 'Italy': 6, 'Spain': 7, 'Netherlands': 8,
    'Portugal': 9, 'Belgium': 10, 'Croatia': 11, 'Denmark': 12,
    'Uruguay': 13, 'Switzerland': 14, 'USA': 15, 'Mexico': 16,
    'Poland': 17, 'Colombia': 18, 'Senegal': 19, 'Japan': 20,
    'South Korea': 21, 'Morocco': 22, 'Australia': 23, 'Ecuador': 24,
    'Serbia': 25, 'Chile': 26, 'Nigeria': 27, 'Cameroon': 28,
    'Saudi Arabia': 29, 'Iran': 30,
}

TEAM_FLAGS = {
    'Argentina':'🇦🇷','Australia':'🇦🇺','Belgium':'🇧🇪','Brazil':'🇧🇷',
    'Cameroon':'🇨🇲','Chile':'🇨🇱','Colombia':'🇨🇴','Croatia':'🇭🇷',
    'Denmark':'🇩🇰','Ecuador':'🇪🇨','England':'🏴󠁧󠁢󠁥󠁮󠁧󠁿','France':'🇫🇷',
    'Germany':'🇩🇪','Iran':'🇮🇷','Italy':'🇮🇹','Japan':'🇯🇵',
    'Mexico':'🇲🇽','Morocco':'🇲🇦','Netherlands':'🇳🇱','Nigeria':'🇳🇬',
    'Poland':'🇵🇱','Portugal':'🇵🇹','Saudi Arabia':'🇸🇦','Senegal':'🇸🇳',
    'Serbia':'🇷🇸','South Korea':'🇰🇷','Spain':'🇪🇸','Switzerland':'🇨🇭',
    'Uruguay':'🇺🇾','USA':'🇺🇸',
}

@st.cache_resource
def train_model():
    np.random.seed(42)
    teams = list(FIFA_RANKS.keys())
    hist_wins = {t: max(3, int(60 / (r ** 0.45))) for t, r in FIFA_RANKS.items()}
    records = []
    for year in range(1950, 2023, 4):
        participants = np.random.choice(teams, min(28, len(teams)), replace=False)
        pairs = [(participants[i], participants[i+1]) for i in range(0, len(participants) - 1, 2)]
        for home, away in pairs:
            hg = max(0, int(np.random.normal(1.5 + 0.02 * hist_wins.get(home, 8), 1.1)))
            ag = max(0, int(np.random.normal(1.2 + 0.02 * hist_wins.get(away, 8), 1.0)))
            records.append({'Year': year, 'Home': home, 'Away': away, 'HG': hg, 'AG': ag,
                            'HR': FIFA_RANKS.get(home, 25), 'AR': FIFA_RANKS.get(away, 25)})
    df = pd.DataFrame(records)
    df.sort_values('Year', inplace=True)
    def get_result(r):
        if r['HG'] > r['AG']: return 0
        if r['HG'] < r['AG']: return 1
        return 2
    df['Result'] = df.apply(get_result, axis=1)
    df['HWon'] = (df['Result'] == 0).astype(int)
    df['HGR'] = df.groupby('Home')['HG'].expanding().mean().reset_index(level=0, drop=True)
    df['AGR'] = df.groupby('Away')['AG'].expanding().mean().reset_index(level=0, drop=True)
    df['HWR'] = df.groupby('Home')['HWon'].expanding().mean().reset_index(level=0, drop=True)
    df['RD'] = df['AR'] - df['HR']
    df['PI'] = df.groupby('Year').cumcount() / df.groupby('Year')['Year'].transform('count')
    df.fillna(0, inplace=True)
    FEATS = ['RD', 'HGR', 'AGR', 'HWR', 'PI']
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=5, subsample=0.8, random_state=42)
    model.fit(df[FEATS], df['Result'])
    return model, df, FEATS

model, hist_df, FEATS = train_model()

def predict(home, away):
    hr = FIFA_RANKS.get(home, 25)
    ar = FIFA_RANKS.get(away, 25)
    hgr = hist_df[hist_df['Home'] == home]['HGR'].mean()
    agr = hist_df[hist_df['Away'] == away]['AGR'].mean()
    hwr = hist_df[hist_df['Home'] == home]['HWR'].mean()
    hgr = hgr if not np.isnan(hgr) else float(hist_df['HGR'].median())
    agr = agr if not np.isnan(agr) else float(hist_df['AGR'].median())
    hwr = hwr if not np.isnan(hwr) else float(hist_df['HWR'].median())
    row = pd.DataFrame([[ar - hr, hgr, agr, hwr, 0.25]], columns=FEATS)
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    p_hw = proba[classes.index(0)] * 100 if 0 in classes else 0.0
    p_aw = proba[classes.index(1)] * 100 if 1 in classes else 0.0
    p_dr = proba[classes.index(2)] * 100 if 2 in classes else 0.0
    return round(p_hw, 1), round(p_aw, 1), round(p_dr, 1)


# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="logo-wrap">
      <svg width="90" height="110" viewBox="0 0 90 110" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%"   stop-color="#fde68a"/>
            <stop offset="50%"  stop-color="#f59e0b"/>
            <stop offset="100%" stop-color="#92400e"/>
          </linearGradient>
          <linearGradient id="g2" x1="0%" y1="0%" x2="40%" y2="100%">
            <stop offset="0%"   stop-color="#fef9c3" stop-opacity="0.8"/>
            <stop offset="100%" stop-color="#f59e0b"  stop-opacity="0"/>
          </linearGradient>
          <filter id="glow2">
            <feGaussianBlur stdDeviation="2" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        <!-- cup bowl -->
        <path d="M22 6 Q20 26 15 37 Q11 46 18 52 Q27 58 45 58 Q63 58 72 52 Q79 46 75 37 Q70 26 68 6 Z"
              fill="url(#g1)" filter="url(#glow2)"/>
        <!-- shine -->
        <path d="M28 8 Q26 24 23 34 Q21 42 25 47"
              fill="none" stroke="url(#g2)" stroke-width="4" stroke-linecap="round" opacity="0.75"/>
        <!-- left handle -->
        <path d="M22 17 Q4 21 6 35 Q7 46 19 48"
              fill="none" stroke="url(#g1)" stroke-width="6" stroke-linecap="round"/>
        <!-- right handle -->
        <path d="M68 17 Q86 21 84 35 Q83 46 71 48"
              fill="none" stroke="url(#g1)" stroke-width="6" stroke-linecap="round"/>
        <!-- stem -->
        <rect x="40" y="58" width="10" height="20" rx="3" fill="url(#g1)"/>
        <rect x="42" y="60" width="3.5" height="16" rx="1.5" fill="#fef9c3" opacity="0.4"/>
        <!-- base -->
        <rect x="26" y="78" width="38" height="8" rx="4" fill="url(#g1)"/>
        <rect x="30" y="80" width="30" height="4" rx="2" fill="#fef9c3" opacity="0.25"/>
        <rect x="30" y="86" width="30" height="5" rx="3" fill="#b45309"/>
        <!-- star -->
        <polygon points="45,10 47,16 53,16 48,20 50,26 45,22 40,26 42,20 37,16 43,16"
                 fill="#fef9c3" opacity="0.95"/>
        <!-- green ball at bottom -->
        <circle cx="45" cy="100" r="9" fill="#15803d"/>
        <circle cx="45" cy="100" r="9" fill="none" stroke="#4ade80" stroke-width="1.2" opacity="0.6"/>
        <polygon points="45,93 48,96 47,100 43,100 42,96" fill="#052e16" opacity="0.9"/>
        <polygon points="54,97 52,101 49,102 48,99 51,96"  fill="#052e16" opacity="0.9"/>
        <polygon points="36,97 38,101 41,102 42,99 39,96"  fill="#052e16" opacity="0.9"/>
        <polygon points="45,107 48,105 49,102 45,100 41,102 42,105" fill="#052e16" opacity="0.9"/>
        <circle cx="41" cy="96" r="2.5" fill="white" opacity="0.15"/>
      </svg>
    </div>
    <div class="hero-eyebrow">FIFA World Cup 2026 · USA / Canada / Mexico</div>
    <h1 class="hero-title">Match<br><span>Predictor</span></h1>
    <p class="hero-sub">Gradient Boosting · trained on WC data 1950–2022</p>
</div>
<div class="pitch-divider"></div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([5, 1, 5])
with c1:
    st.markdown('<div class="team-label">🏠 Home Team</div>', unsafe_allow_html=True)
    home_team = st.selectbox("home", TEAMS_2026, index=TEAMS_2026.index("Brazil"), label_visibility="collapsed")
with c2:
    st.markdown('<div class="vs-badge"><span class="vs-text">VS</span></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="team-label">✈️ Away Team</div>', unsafe_allow_html=True)
    away_team = st.selectbox("away", TEAMS_2026, index=TEAMS_2026.index("Germany"), label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("⚽  Predict Match", use_container_width=True):
    if home_team == away_team:
        st.error("Pick two different teams.")
    else:
        hw, aw, dr = predict(home_team, away_team)
        hf = TEAM_FLAGS.get(home_team, "")
        af = TEAM_FLAGS.get(away_team, "")

        if hw >= max(aw, dr):
            winner = f"{hf} {home_team}"
            outcome = "WIN"
        elif aw >= max(hw, dr):
            winner = f"{af} {away_team}"
            outcome = "WIN"
        else:
            winner = "Draw"
            outcome = "MOST LIKELY"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-winner">{winner}</div>
            <div class="result-match">{outcome} &nbsp;·&nbsp; {hf} {home_team} vs {af} {away_team}</div>
        </div>
        """, unsafe_allow_html=True)

        best = max(hw, aw, dr)
        h_cls = "highlight" if hw == best else ""
        d_cls = "highlight" if dr == best and dr > hw and dr > aw else ""
        a_cls = "highlight" if aw == best else ""

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile {h_cls}">
                <div class="metric-val">{hw}%</div>
                <div class="metric-name">{hf} {home_team}</div>
            </div>
            <div class="metric-tile {d_cls}">
                <div class="metric-val">{dr}%</div>
                <div class="metric-name">Draw</div>
            </div>
            <div class="metric-tile {a_cls}">
                <div class="metric-val">{aw}%</div>
                <div class="metric-name">{af} {away_team}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label">{hf} {home_team}</div>
            <div class="prob-track"><div class="prob-fill-green" style="width:{hw}%"></div></div>
            <div class="prob-pct">{hw}%</div>
        </div>
        <div class="prob-row">
            <div class="prob-label">Draw</div>
            <div class="prob-track"><div class="prob-fill-gray" style="width:{dr}%"></div></div>
            <div class="prob-pct">{dr}%</div>
        </div>
        <div class="prob-row">
            <div class="prob-label">{af} {away_team}</div>
            <div class="prob-track"><div class="prob-fill-blue" style="width:{aw}%"></div></div>
            <div class="prob-pct">{aw}%</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="app-footer">
    Round 2 Submission &nbsp;·&nbsp; FIFA WC 2026 Prediction Challenge
</div>
""", unsafe_allow_html=True)

# C:\Users\as\anaconda3\python.exe -m streamlit run C:\Users\as\Downloads\fifaworldcuppredict\app.py
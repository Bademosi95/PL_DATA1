import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="⚽ Football Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .stSelectbox > div > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

class FootballPredictor:
    def __init__(self):
        self.df = None
        self.league_avg_home_goals = None
        self.league_avg_away_goals = None
        
    @st.cache_data
    def load_data(_self, url=None):
        """Load data from URL with caching"""
        try:
            if url:
                df = pd.read_html(url, attrs={"id":"results2025-202691_home_away_sh"})[0]
            else:
                # Fallback sample data if URL fails
                sample_teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 
                              'Tottenham', 'Newcastle', 'Brighton', 'Aston Villa', 'West Ham']
                df = pd.DataFrame({
                    ('Unnamed: 0_level_0', 'Rk'): range(1, 11),
                    ('Unnamed: 1_level_0', 'Squad'): sample_teams,
                    ('Home', 'MP'): np.random.randint(15, 20, 10),
                    ('Home', 'GF'): np.random.randint(20, 40, 10),
                    ('Home', 'GA'): np.random.randint(15, 35, 10),
                    ('Away', 'MP'): np.random.randint(15, 20, 10),
                    ('Away', 'GF'): np.random.randint(15, 35, 10),
                    ('Away', 'GA'): np.random.randint(20, 40, 10),
                })
            
            # Clean column names
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            df.rename(columns={df.columns[1]: "Team"}, inplace=True)
            df.set_index("Team", inplace=True)
            
            # Calculate league averages
            league_avg_home_goals = df["Home_GF"].sum() / df["Home_MP"].sum()
            league_avg_away_goals = df["Away_GF"].sum() / df["Away_MP"].sum()
            
            return df, league_avg_home_goals, league_avg_away_goals, True
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, None, False
    
    def calculate_team_strengths(self, team, df, league_avg_home_goals, league_avg_away_goals):
        """Calculate team strength metrics"""
        try:
            home_attack = df.loc[team, "Home_GF"] / df.loc[team, "Home_MP"] / league_avg_home_goals
            home_defense = df.loc[team, "Home_GA"] / df.loc[team, "Home_MP"] / league_avg_away_goals
            away_attack = df.loc[team, "Away_GF"] / df.loc[team, "Away_MP"] / league_avg_away_goals
            away_defense = df.loc[team, "Away_GA"] / df.loc[team, "Away_MP"] / league_avg_home_goals
            return home_attack, home_defense, away_attack, away_defense
        except KeyError:
            st.error(f"Team '{team}' not found in dataset")
            return None, None, None, None
    
    def dixon_coles_adjustment(self, home_goals, away_goals, rho=0.1):
        """Dixon-Coles adjustment for low-scoring games"""
        if home_goals == 0 and away_goals == 0:
            return 1 - rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - rho
        else:
            return 1.0
    
    def enhanced_poisson_prediction(self, home_team, away_team, df, league_avg_home_goals, league_avg_away_goals,
                                  home_advantage=0.05, injury_handicap_home=0.0, injury_handicap_away=0.0, use_dixon_coles=True):
        """Enhanced prediction with Dixon-Coles adjustment"""
        
        home_strengths = self.calculate_team_strengths(home_team, df, league_avg_home_goals, league_avg_away_goals)
        away_strengths = self.calculate_team_strengths(away_team, df, league_avg_home_goals, league_avg_away_goals)
        
        if None in home_strengths or None in away_strengths:
            return None
        
        h_attack, h_def, h_away_attack, h_away_def = home_strengths
        a_attack, a_def, a_away_attack, a_away_def = away_strengths
        
        # Calculate expected goals
        home_goals_avg = (league_avg_home_goals * h_attack * a_def * 
                         (1 + home_advantage - injury_handicap_home))
        away_goals_avg = (league_avg_away_goals * a_away_attack * h_def * 
                         (1 - injury_handicap_away))
        
        # Ensure positive values
        home_goals_avg = max(0.1, home_goals_avg)
        away_goals_avg = max(0.1, away_goals_avg)
        
        max_goals = 8
        probs = np.zeros((max_goals, max_goals))
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob = poisson.pmf(i, home_goals_avg) * poisson.pmf(j, away_goals_avg)
                if use_dixon_coles:
                    prob *= self.dixon_coles_adjustment(i, j)
                probs[i, j] = prob
        
        # Normalize probabilities
        probs = probs / np.sum(probs)
        
        # Calculate outcome probabilities
        home_win_prob = np.sum(np.tril(probs, -1))
        draw_prob = np.sum(np.diag(probs))
        away_win_prob = np.sum(np.triu(probs, 1))
        
        # BTTS probability
        btts_prob = 1 - probs[0, :].sum() - probs[:, 0].sum() + probs[0, 0]
        
        # Over/Under 2.5 goals
        over_2_5_prob = np.sum([probs[i, j] for i in range(max_goals) 
                               for j in range(max_goals) if i + j > 2.5])
        
        # Expected goals
        expected_home_goals = np.sum([i * probs[i, :].sum() for i in range(max_goals)])
        expected_away_goals = np.sum([j * probs[:, j].sum() for j in range(max_goals)])
        
        return {
            "home_win": home_win_prob,
            "draw": draw_prob,
            "away_win": away_win_prob,
            "btts": btts_prob,
            "over_2_5": over_2_5_prob,
            "expected_home_goals": expected_home_goals,
            "expected_away_goals": expected_away_goals,
            "prob_matrix": probs,
            "home_goals_lambda": home_goals_avg,
            "away_goals_lambda": away_goals_avg
        }

def create_outcome_chart(result, home_team, away_team):
    """Create outcome probability chart"""
    outcomes = ['Home Win', 'Draw', 'Away Win']
    probabilities = [result['home_win'], result['draw'], result['away_win']]
    colors = ['#2E8B57', '#FFD700', '#DC143C']
    
    fig = go.Figure(data=[
        go.Bar(x=outcomes, y=probabilities, 
               marker_color=colors,
               text=[f'{p:.1%}' for p in probabilities],
               textposition='auto',
               hovertemplate='%{x}: %{y:.1%}<extra></extra>')
    ])
    
    fig.update_layout(
        title=f'Match Outcome Probabilities<br>{home_team} vs {away_team}',
        yaxis_title='Probability',
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_score_matrix(result):
    """Create score probability heatmap"""
    matrix = result['prob_matrix'][:6, :6]  # Show 6x6 grid
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=list(range(6)),
        y=list(range(6)),
        colorscale='Blues',
        text=[[f'{matrix[i, j]:.3f}' for j in range(6)] for i in range(6)],
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Home: %{y} - Away: %{x}<br>Probability: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Score Probability Matrix',
        xaxis_title='Away Team Goals',
        yaxis_title='Home Team Goals',
        height=400
    )
    
    return fig

def create_special_markets_chart(result):
    """Create special markets chart"""
    markets = ['BTTS', 'Over 2.5 Goals']
    probabilities = [result['btts'], result['over_2_5']]
    colors = ['#FF6347', '#9370DB']
    
    fig = go.Figure(data=[
        go.Bar(x=markets, y=probabilities,
               marker_color=colors,
               text=[f'{p:.1%}' for p in probabilities],
               textposition='auto',
               hovertemplate='%{x}: %{y:.1%}<extra></extra>')
    ])
    
    fig.update_layout(
        title='Special Markets',
        yaxis_title='Probability',
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_expected_goals_chart(result, home_team, away_team):
    """Create expected goals chart"""
    teams = [home_team[:15], away_team[:15]]  # Truncate long names
    exp_goals = [result['expected_home_goals'], result['expected_away_goals']]
    colors = ['#4CAF50', '#FF5722']
    
    fig = go.Figure(data=[
        go.Bar(x=teams, y=exp_goals,
               marker_color=colors,
               text=[f'{g:.2f}' for g in exp_goals],
               textposition='auto',
               hovertemplate='%{x}: %{y:.2f} goals<extra></extra>')
    ])
    
    fig.update_layout(
        title='Expected Goals',
        yaxis_title='Goals',
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ Football Match Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize predictor
    predictor = FootballPredictor()
    
    # Sidebar for inputs
    st.sidebar.header("🎯 Match Prediction Settings")
    
    # Load data
    with st.spinner("Loading team data..."):
        url = 'https://fbref.com/en/comps/9/Premier-League-Stats'
        df, league_avg_home_goals, league_avg_away_goals, success = predictor.load_data(url)
    
    if not success or df is None:
        st.error("Failed to load data. Please check your internet connection or try again later.")
        st.stop()
    
    # Team selection
    teams = sorted(df.index.tolist())
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home Team", teams, key="home_team")
    with col2:
        away_team = st.selectbox("✈️ Away Team", teams, key="away_team")
    
    # Injury/Form handicaps
    st.sidebar.subheader("🏥 Team Conditions")
    
    injury_handicap_home = st.sidebar.slider(
        "Home Team Handicap",
        min_value=0.0,
        max_value=0.3,
        value=0.0,
        step=0.05,
        help="Injury/suspension/poor form impact on home team (0 = no impact, 0.3 = severe impact)"
    )
    
    injury_handicap_away = st.sidebar.slider(
        "Away Team Handicap",
        min_value=0.0,
        max_value=0.3,
        value=0.0,
        step=0.05,
        help="Injury/suspension/poor form impact on away team (0 = no impact, 0.3 = severe impact)"
    )
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        home_advantage = st.slider("Home Advantage", 0.0, 0.2, 0.05, 0.01)
        use_dixon_coles = st.checkbox("Use Dixon-Coles Adjustment", True)
    
    # Prediction button
    if st.sidebar.button("🔮 Generate Prediction", type="primary", use_container_width=True):
        if home_team == away_team:
            st.error("Please select different teams for home and away.")
        else:
            with st.spinner("Calculating prediction..."):
                result = predictor.enhanced_poisson_prediction(
                    home_team, away_team, df, league_avg_home_goals, league_avg_away_goals,
                    home_advantage=home_advantage,
                    injury_handicap_home=injury_handicap_home,
                    injury_handicap_away=injury_handicap_away,
                    use_dixon_coles=use_dixon_coles
                )
            
            if result:
                # Main prediction display
                st.subheader(f"🎯 Prediction: {home_team} vs {away_team}")
                
                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🏠 Home Win",
                        f"{result['home_win']:.1%}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "🤝 Draw",
                        f"{result['draw']:.1%}",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "✈️ Away Win",
                        f"{result['away_win']:.1%}",
                        delta=None
                    )
                
                with col4:
                    most_likely = max([
                        ("Home Win", result['home_win']),
                        ("Draw", result['draw']),
                        ("Away Win", result['away_win'])
                    ], key=lambda x: x[1])
                    
                    st.metric(
                        "🏆 Most Likely",
                        most_likely[0],
                        f"{most_likely[1]:.1%}"
                    )
                
                # Charts
                st.subheader("📊 Detailed Analysis")
                
                # First row of charts
                col1, col2 = st.columns(2)
                
                with col1:
                    outcome_fig = create_outcome_chart(result, home_team, away_team)
                    st.plotly_chart(outcome_fig, use_container_width=True)
                
                with col2:
                    exp_goals_fig = create_expected_goals_chart(result, home_team, away_team)
                    st.plotly_chart(exp_goals_fig, use_container_width=True)
                
                # Second row of charts
                col1, col2 = st.columns(2)
                
                with col1:
                    special_markets_fig = create_special_markets_chart(result)
                    st.plotly_chart(special_markets_fig, use_container_width=True)
                
                with col2:
                    score_matrix_fig = create_score_matrix(result)
                    st.plotly_chart(score_matrix_fig, use_container_width=True)
                
                # Additional metrics
                st.subheader("📈 Additional Insights")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "⚽ Expected Goals (Home)",
                        f"{result['expected_home_goals']:.2f}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "⚽ Expected Goals (Away)",
                        f"{result['expected_away_goals']:.2f}",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "🎯 Both Teams Score",
                        f"{result['btts']:.1%}",
                        delta=None
                    )
                
                with col4:
                    st.metric(
                        "🔥 Over 2.5 Goals",
                        f"{result['over_2_5']:.1%}",
                        delta=None
                    )
                
                # Technical details in expander
                with st.expander("🔧 Technical Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Home Team λ (Poisson parameter):** {result['home_goals_lambda']:.3f}")
                        st.write(f"**Home Advantage Applied:** {home_advantage:.1%}")
                        st.write(f"**Home Team Handicap:** {injury_handicap_home:.1%}")
                    
                    with col2:
                        st.write(f"**Away Team λ (Poisson parameter):** {result['away_goals_lambda']:.3f}")
                        st.write(f"**Dixon-Coles Adjustment:** {'Yes' if use_dixon_coles else 'No'}")
                        st.write(f"**Away Team Handicap:** {injury_handicap_away:.1%}")
            else:
                st.error("Failed to generate prediction. Please check team selections.")
    
    # Team statistics
    with st.expander("📊 Team Statistics"):
        if df is not None:
            # Create team stats summary
            stats_df = pd.DataFrame(index=df.index)
            stats_df['Games Played'] = df['Home_MP'] + df['Away_MP']
            stats_df['Goals For'] = df['Home_GF'] + df['Away_GF']
            stats_df['Goals Against'] = df['Home_GA'] + df['Away_GA']
            stats_df['Goal Difference'] = stats_df['Goals For'] - stats_df['Goals Against']
            stats_df['Goals Per Game'] = stats_df['Goals For'] / stats_df['Games Played']
            stats_df['Home Goals/Game'] = df['Home_GF'] / df['Home_MP']
            stats_df['Away Goals/Game'] = df['Away_GF'] / df['Away_MP']
            
            st.dataframe(stats_df.round(2), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            ⚽ KB Football Match Predictor<br>
            Predictions are for entertainment purposes only. Past performance does not guarantee future results.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
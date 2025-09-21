#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FootballPredictor:
    def __init__(self):
        self.df = None
        self.league_avg_home_goals = None
        self.league_avg_away_goals = None
        self.team_form = {}
        self.historical_data = None
        
    def load_data(self, url=None, csv_path=None):
        """Load data from URL or CSV with error handling"""
        try:
            if url:
                self.df = pd.read_html(url, attrs={"id":"results2025-202691_home_away_sh"})[0]
            elif csv_path:
                self.df = pd.read_csv(csv_path)
            else:
                raise ValueError("Either URL or CSV path must be provided")
                
            # Clean column names
            self.df.columns = ['_'.join(col).strip() for col in self.df.columns.values]
            self.df.rename(columns={self.df.columns[1]: "Team"}, inplace=True)
            self.df.set_index("Team", inplace=True)
            
            # Calculate league averages
            self.league_avg_home_goals = self.df["Home_GF"].sum() / self.df["Home_MP"].sum()
            self.league_avg_away_goals = self.df["Away_GF"].sum() / self.df["Away_MP"].sum()
            
            print(f"Data loaded successfully. {len(self.df)} teams found.")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def calculate_team_strengths_advanced(self, team):
        """Enhanced team strength calculation with weighted metrics"""
        try:
            # Basic strength metrics
            home_attack = self.df.loc[team, "Home_GF"] / self.df.loc[team, "Home_MP"] / self.league_avg_home_goals
            home_defense = self.df.loc[team, "Home_GA"] / self.df.loc[team, "Home_MP"] / self.league_avg_away_goals
            away_attack = self.df.loc[team, "Away_GF"] / self.df.loc[team, "Away_MP"] / self.league_avg_away_goals
            away_defense = self.df.loc[team, "Away_GA"] / self.df.loc[team, "Away_MP"] / self.league_avg_home_goals
            
            # Add form factor if available
            form_factor = self.team_form.get(team, 1.0)
            
            return {
                'home_attack': home_attack * form_factor,
                'home_defense': home_defense / form_factor,  # Better form = better defense
                'away_attack': away_attack * form_factor,
                'away_defense': away_defense / form_factor
            }
        except KeyError:
            print(f"Team '{team}' not found in dataset")
            return None
    
    def update_team_form(self, team, recent_form_multiplier):
        """Update team form based on recent performance"""
        self.team_form[team] = recent_form_multiplier
    
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
    
    def enhanced_poisson_prediction(self, home_team, away_team, 
                                  home_advantage=0.05, 
                                  injury_handicap_home=0.0, 
                                  injury_handicap_away=0.0,
                                  use_dixon_coles=True):
        """Enhanced prediction with Dixon-Coles adjustment and better error handling"""
        
        home_strengths = self.calculate_team_strengths_advanced(home_team)
        away_strengths = self.calculate_team_strengths_advanced(away_team)
        
        if not home_strengths or not away_strengths:
            return None
        
        # Calculate expected goals with adjustments
        home_goals_avg = (self.league_avg_home_goals * 
                         home_strengths['home_attack'] * 
                         away_strengths['away_defense'] * 
                         (1 + home_advantage - injury_handicap_home))
        
        away_goals_avg = (self.league_avg_away_goals * 
                         away_strengths['away_attack'] * 
                         home_strengths['home_defense'] * 
                         (1 - injury_handicap_away))
        
        # Ensure positive values
        home_goals_avg = max(0.1, home_goals_avg)
        away_goals_avg = max(0.1, away_goals_avg)
        
        max_goals = 8  # Increased for better accuracy
        probs = np.zeros((max_goals, max_goals))
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob = poisson.pmf(i, home_goals_avg) * poisson.pmf(j, away_goals_avg)
                
                # Apply Dixon-Coles adjustment
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
        
        # Expected goals (more accurate calculation)
        expected_home_goals = np.sum([i * probs[i, :].sum() for i in range(max_goals)])
        expected_away_goals = np.sum([j * probs[:, j].sum() for j in range(max_goals)])
        
        return {
            "home_win": round(home_win_prob, 4),
            "draw": round(draw_prob, 4),
            "away_win": round(away_win_prob, 4),
            "btts": round(btts_prob, 4),
            "over_2_5": round(over_2_5_prob, 4),
            "expected_home_goals": round(expected_home_goals, 2),
            "expected_away_goals": round(expected_away_goals, 2),
            "prob_matrix": probs,
            "home_goals_lambda": round(home_goals_avg, 2),
            "away_goals_lambda": round(away_goals_avg, 2)
        }
    
    def batch_predictions(self, fixtures_list):
        """Generate predictions for multiple fixtures"""
        predictions = []
        for home_team, away_team in fixtures_list:
            pred = self.enhanced_poisson_prediction(home_team, away_team)
            if pred:
                pred['fixture'] = f"{home_team} vs {away_team}"
                predictions.append(pred)
        return pd.DataFrame(predictions)
    
    def visualize_prediction(self, home_team, away_team):
        """Create visualization of prediction probabilities"""
        result = self.enhanced_poisson_prediction(home_team, away_team)
        if not result:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Outcome probabilities
        outcomes = ['Home Win', 'Draw', 'Away Win']
        probs = [result['home_win'], result['draw'], result['away_win']]
        colors = ['green', 'gray', 'red']
        
        ax1.bar(outcomes, probs, color=colors, alpha=0.7)
        ax1.set_title('Match Outcome Probabilities')
        ax1.set_ylabel('Probability')
        for i, v in enumerate(probs):
            ax1.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
        
        # 2. Score probability matrix
        im = ax2.imshow(result['prob_matrix'][:6, :6], cmap='Blues', origin='lower')
        ax2.set_xlabel('Away Goals')
        ax2.set_ylabel('Home Goals')
        ax2.set_title('Score Probability Matrix')
        
        # Add text annotations
        for i in range(6):
            for j in range(6):
                text = ax2.text(j, i, f'{result["prob_matrix"][i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        # 3. Special markets
        markets = ['BTTS', 'Over 2.5']
        market_probs = [result['btts'], result['over_2_5']]
        
        ax3.bar(markets, market_probs, color=['orange', 'purple'], alpha=0.7)
        ax3.set_title('Special Markets')
        ax3.set_ylabel('Probability')
        for i, v in enumerate(market_probs):
            ax3.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
        
        # 4. Expected goals
        teams = [home_team[:10], away_team[:10]]  # Truncate long names
        exp_goals = [result['expected_home_goals'], result['expected_away_goals']]
        
        bars = ax4.bar(teams, exp_goals, color=['lightblue', 'lightcoral'], alpha=0.7)
        ax4.set_title('Expected Goals')
        ax4.set_ylabel('Goals')
        for bar, goal in zip(bars, exp_goals):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{goal:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def model_validation(self, historical_results):
        """Validate model against historical results"""
        if historical_results is None or len(historical_results) == 0:
            print("No historical data provided for validation")
            return
        
        predictions = []
        actuals = []
        
        for _, match in historical_results.iterrows():
            pred = self.enhanced_poisson_prediction(match['home_team'], match['away_team'])
            if pred:
                # Convert to outcome probabilities
                pred_probs = [pred['home_win'], pred['draw'], pred['away_win']]
                predictions.append(pred_probs)
                
                # Actual outcome (1-hot encoded)
                if match['home_goals'] > match['away_goals']:
                    actuals.append([1, 0, 0])
                elif match['home_goals'] < match['away_goals']:
                    actuals.append([0, 0, 1])
                else:
                    actuals.append([0, 1, 0])
        
        if predictions:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Calculate log loss (lower is better)
            loss = log_loss(actuals, predictions)
            
            # Calculate accuracy
            pred_outcomes = np.argmax(predictions, axis=1)
            actual_outcomes = np.argmax(actuals, axis=1)
            accuracy = accuracy_score(actual_outcomes, pred_outcomes)
            
            print(f"Model Validation Results:")
            print(f"Log Loss: {loss:.4f}")
            print(f"Accuracy: {accuracy:.2%}")
            
            return {'log_loss': loss, 'accuracy': accuracy}
    
    def get_team_stats_summary(self):
        """Get comprehensive team statistics"""
        if self.df is None:
            return None
        
        summary = pd.DataFrame(index=self.df.index)
        summary['Total_Games'] = self.df['Home_MP'] + self.df['Away_MP']
        summary['Total_Goals_For'] = self.df['Home_GF'] + self.df['Away_GF']
        summary['Total_Goals_Against'] = self.df['Home_GA'] + self.df['Away_GA']
        summary['Goal_Difference'] = summary['Total_Goals_For'] - summary['Total_Goals_Against']
        summary['Goals_Per_Game'] = summary['Total_Goals_For'] / summary['Total_Games']
        summary['Goals_Conceded_Per_Game'] = summary['Total_Goals_Against'] / summary['Total_Games']
        
        # Home and away splits
        summary['Home_Goals_Per_Game'] = self.df['Home_GF'] / self.df['Home_MP']
        summary['Away_Goals_Per_Game'] = self.df['Away_GF'] / self.df['Away_MP']
        
        return summary.round(2)

# Example usage
def main():
    predictor = FootballPredictor()
    
    # Load data
    url = 'https://fbref.com/en/comps/9/Premier-League-Stats'
    if predictor.load_data(url=url):
        
        # Example predictions
        home_team = "Arsenal"
        away_team = "Manchester City"
        
        # Basic prediction
        result = predictor.enhanced_poisson_prediction(home_team, away_team)
        if result:
            print(f"\nPrediction for {home_team} vs {away_team}:")
            print(f"Home Win: {result['home_win']:.1%}")
            print(f"Draw: {result['draw']:.1%}")
            print(f"Away Win: {result['away_win']:.1%}")
            print(f"BTTS: {result['btts']:.1%}")
            print(f"Over 2.5: {result['over_2_5']:.1%}")
            print(f"Expected Goals: {result['expected_home_goals']} - {result['expected_away_goals']}")
        
        # Team statistics
        stats = predictor.get_team_stats_summary()
        if stats is not None:
            print(f"\nTop 5 attacking teams:")
            print(stats.nlargest(5, 'Goals_Per_Game')[['Goals_Per_Game', 'Goals_Conceded_Per_Game']])

if __name__ == "__main__":
    main()


# In[ ]:





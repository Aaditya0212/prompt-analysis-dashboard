import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class SimpleDatasetCleaner: 
    def __init__(self, input_file="enhanced_prompt_logs.csv"):
        self.input_file = input_file
        self.df = None
        print("=" * 40)
    
    def load_and_clean_data(self):
        print("Loading your dataset")
        
        try:
            # Load the dataset
            self.df = pd.read_csv(self.input_file)
            print(f"Loaded {len(self.df)} records")
            
            # Convert timestamp
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            # Fill missing values
            self.df['prompt_text'] = self.df['prompt_text'].fillna('')
            self.df['error_description'] = self.df['error_description'].fillna('No Error')
            self.df['feedback_rating'] = self.df['feedback_rating'].fillna(self.df['feedback_rating'].median())
            
            # Clean prompt text
            self.df['prompt_text_clean'] = self.df['prompt_text'].apply(self._clean_text)
            
            print("Data cleaned successfully")
            return self.df
            
        except FileNotFoundError:
            print(f"Error: File not found: {self.input_file}")
            raise
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def _clean_text(self, text):
        if pd.isna(text):
            return ""
        
        # Convert to string and clean
        text = str(text).strip().lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def create_time_features(self):
        print("Creating time features")
        
        # Extract time components
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['day_name'] = self.df['timestamp'].dt.day_name()
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['month_name'] = self.df['timestamp'].dt.month_name()
        self.df['year'] = self.df['timestamp'].dt.year
        self.df['date'] = self.df['timestamp'].dt.date
        
        # Time periods
        def get_time_period(hour):
            if 0 <= hour < 6:
                return 'Night'
            elif 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 22:
                return 'Evening'
            else:
                return 'Late Night'
        
        self.df['time_period'] = self.df['hour'].apply(get_time_period)
        
        # Business indicators
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])
        self.df['is_business_hours'] = self.df['hour'].between(9, 17)
        
        print("Time features created")
    
    def create_text_features(self):
        print("Creating text features")
        
        self.df['word_count'] = self.df['prompt_text_clean'].apply(
            lambda x: len(str(x).split()) if x else 0
        )
        self.df['char_count'] = self.df['prompt_text_clean'].apply(len)
        
        # Question detection
        self.df['has_question'] = self.df['prompt_text_clean'].str.contains(r'\?', na=False)
        self.df['question_count'] = self.df['prompt_text_clean'].str.count(r'\?')
        
        # Code-related detection
        code_keywords = ['function', 'class', 'import', 'return', 'python', 'javascript', 'sql', 'code']
        code_pattern = '|'.join(code_keywords)
        self.df['has_code_keywords'] = self.df['prompt_text_clean'].str.contains(
            code_pattern, case=False, na=False
        )
        
        # Help/learning detection
        help_keywords = ['help', 'explain', 'how to', 'what is', 'guide', 'tutorial']
        help_pattern = '|'.join(help_keywords)
        self.df['is_help_request'] = self.df['prompt_text_clean'].str.contains(
            help_pattern, case=False, na=False
        )
        
        # Creative request detection
        creative_keywords = ['write', 'create', 'generate', 'make', 'design', 'story']
        creative_pattern = '|'.join(creative_keywords)
        self.df['is_creative_request'] = self.df['prompt_text_clean'].str.contains(
            creative_pattern, case=False, na=False
        )
        
        print("Text features created")
    
    def create_business_metrics(self):
        print("Creating business metrics")
        
        # Success indicators
        self.df['is_successful'] = (self.df['completion_status'] == 'Success').astype(int)
        self.df['has_error'] = (~self.df['completion_status'].isin(['Success'])).astype(int)
        
        # Quality indicators
        self.df['is_high_rating'] = (self.df['feedback_rating'] >= 4).astype(int)
        self.df['is_low_rating'] = (self.df['feedback_rating'] <= 2).astype(int)
        
        # Performance indicators
        self.df['is_fast_response'] = (self.df['response_time'] <= 3).astype(int)
        self.df['is_slow_response'] = (self.df['response_time'] > 10).astype(int)
        
        # User type indicators
        self.df['is_premium'] = self.df['is_premium_user'].astype(int)
        
        # Version indicators
        self.df['is_version_v1'] = (self.df['prompt_version'] == 'v1').astype(int)
        self.df['is_version_v2'] = (self.df['prompt_version'] == 'v2').astype(int)
        
        # Sentiment categories (simple)
        def categorize_sentiment(score):
            if score > 0.1:
                return 'Positive'
            elif score < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        
        self.df['sentiment_category'] = self.df['sentiment_score'].apply(categorize_sentiment)
        self.df['is_positive_sentiment'] = (self.df['sentiment_category'] == 'Positive').astype(int)
        
        # Engagement score (simple calculation)
        self.df['engagement_score'] = (
            self.df['feedback_rating'] * 0.4 +
            self.df['is_successful'] * 2 * 0.3 +
            (self.df['word_count'] / 20).clip(0, 2) * 0.3
        ).round(2)
        
        print("Business metrics created")
    
    def create_user_sessions(self):
        print("Creating user sessions")
        
        # Sort by user and timestamp
        df_sorted = self.df.sort_values(['user_id', 'timestamp']).copy()
        
        # Calculate time differences between prompts
        df_sorted['time_diff_minutes'] = df_sorted.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 60
        
        # Mark new sessions (30+ minute gaps)
        df_sorted['is_new_session'] = (
            (df_sorted['time_diff_minutes'] > 30) | 
            (df_sorted['time_diff_minutes'].isna())
        )
        
        # Create session IDs
        df_sorted['session_number'] = df_sorted.groupby('user_id')['is_new_session'].cumsum()
        df_sorted['session_id'] = (
            df_sorted['user_id'].astype(str) + '_session_' + 
            df_sorted['session_number'].astype(str)
        )
        
        # Calculate session-level metrics
        session_stats = df_sorted.groupby('session_id').agg({
            'prompt_id': 'count',
            'timestamp': ['min', 'max'],
            'feedback_rating': 'mean',
            'is_successful': 'mean',
            'user_id': 'first'
        })
        
        # Flatten column names
        session_stats.columns = ['prompts_in_session', 'session_start', 'session_end', 
                                'avg_session_rating', 'session_success_rate', 'user_id']
        
        # Calculate session duration
        session_stats['session_duration_minutes'] = (
            session_stats['session_end'] - session_stats['session_start']
        ).dt.total_seconds() / 60
        session_stats['session_duration_minutes'] = session_stats['session_duration_minutes'].fillna(0)
        
        # Session type classification
        def classify_session(prompt_count):
            if prompt_count == 1:
                return 'Single Query'
            elif prompt_count <= 3:
                return 'Short Session'
            elif prompt_count <= 6:
                return 'Medium Session'
            else:
                return 'Long Session'
        
        session_stats['session_type'] = session_stats['prompts_in_session'].apply(classify_session)
        
        # Merge session info back to main dataframe
        session_info = session_stats[['prompts_in_session', 'session_duration_minutes', 
                                    'session_type', 'avg_session_rating']].reset_index()
        
        # Add session info to main dataframe
        prompt_to_session = df_sorted[['prompt_id', 'session_id']].set_index('prompt_id')
        self.df = self.df.merge(prompt_to_session, left_on='prompt_id', right_index=True, how='left')
        self.df = self.df.merge(session_info, on='session_id', how='left')
        
        # Fill any missing session data
        self.df['prompts_in_session'] = self.df['prompts_in_session'].fillna(1)
        self.df['session_duration_minutes'] = self.df['session_duration_minutes'].fillna(0)
        self.df['session_type'] = self.df['session_type'].fillna('Single Query')
        self.df['avg_session_rating'] = self.df['avg_session_rating'].fillna(self.df['feedback_rating'])
        
        print("User sessions created")
    
    def create_simple_ab_analysis(self):
        print("Creating A/B testing analysis")
        
        # Group by version
        v1_stats = self.df[self.df['prompt_version'] == 'v1'].agg({
            'feedback_rating': 'mean',
            'is_successful': 'mean',
            'response_time': 'mean',
            'engagement_score': 'mean'
        }).round(3)
        
        v2_stats = self.df[self.df['prompt_version'] == 'v2'].agg({
            'feedback_rating': 'mean',
            'is_successful': 'mean',
            'response_time': 'mean',
            'engagement_score': 'mean'
        }).round(3)
        
        # Calculate improvements
        rating_improvement = ((v2_stats['feedback_rating'] - v1_stats['feedback_rating']) / v1_stats['feedback_rating']) * 100
        success_improvement = ((v2_stats['is_successful'] - v1_stats['is_successful']) / v1_stats['is_successful']) * 100
        
        # Add A/B testing results to dataframe
        self.df['v1_avg_rating'] = v1_stats['feedback_rating']
        self.df['v2_avg_rating'] = v2_stats['feedback_rating']
        self.df['rating_improvement_pct'] = rating_improvement
        
        self.df['v1_success_rate'] = v1_stats['is_successful']
        self.df['v2_success_rate'] = v2_stats['is_successful']
        self.df['success_improvement_pct'] = success_improvement
        
        # Determine winner
        if rating_improvement > 0 and success_improvement > 0:
            winner = 'V2 Wins'
        elif rating_improvement < 0 and success_improvement < 0:
            winner = 'V1 Wins'
        else:
            winner = 'Mixed Results'
        
        self.df['ab_test_result'] = winner
        
        print(f"Rating improvement: {rating_improvement:+.1f}%")
        print(f"Success improvement: {success_improvement:+.1f}%")
        print(f"Winner: {winner}")
        
        print("A/B testing analysis created")
    
    def create_user_insights(self):
        print("Creating user insights")
        
        # User-level aggregations
        user_stats = self.df.groupby('user_id').agg({
            'prompt_id': 'count',
            'is_successful': 'mean',
            'feedback_rating': 'mean',
            'engagement_score': 'mean',
            'prompts_in_session': 'mean',
            'user_segment': 'first',
            'is_premium_user': 'first'
        }).round(3)
        
        # User type classification
        def classify_user_type(prompt_count):
            if prompt_count >= 10:
                return 'Power User'
            elif prompt_count >= 3:
                return 'Regular User'
            else:
                return 'Casual User'
        
        user_stats['user_type'] = user_stats['prompt_id'].apply(classify_user_type)
        
        # Add user insights back to main dataframe
        user_lookup = user_stats[['user_type']].reset_index()
        self.df = self.df.merge(user_lookup, on='user_id', how='left')
        
        # User value indicators
        self.df['is_power_user'] = (self.df['user_type'] == 'Power User').astype(int)
        self.df['is_regular_user'] = (self.df['user_type'] == 'Regular User').astype(int)
        
        print("User insights created")
    
    def create_final_dataset(self):
        print("Creating final Power BI dataset")
        
        # Select columns for Power BI (clean and organized)
        powerbi_columns = [
            # Core identifiers
            'prompt_id', 'user_id', 'session_id', 'timestamp', 'date',
            
            # Basic information
            'prompt_type', 'user_segment', 'prompt_version', 'device_type', 'location',
            'is_premium_user', 'user_type',
            
            # Performance metrics
            'feedback_rating', 'response_time', 'completion_status', 
            'is_successful', 'has_error', 'is_high_rating', 'is_fast_response',
            
            # Time analysis
            'hour', 'day_of_week', 'day_name', 'month', 'month_name',
            'time_period', 'is_weekend', 'is_business_hours',
            
            # Text analysis
            'word_count', 'char_count', 'has_question', 'question_count',
            'has_code_keywords', 'is_help_request', 'is_creative_request',
            
            # Sentiment analysis
            'sentiment_score', 'sentiment_category', 'is_positive_sentiment',
            
            # Business metrics
            'engagement_score', 'is_premium', 'is_power_user',
            
            # Session analysis
            'prompts_in_session', 'session_duration_minutes', 'session_type', 'avg_session_rating',
            
            # A/B testing
            'is_version_v1', 'is_version_v2', 'ab_test_result',
            'v1_avg_rating', 'v2_avg_rating', 'rating_improvement_pct',
            'v1_success_rate', 'v2_success_rate', 'success_improvement_pct'
        ]
        
        # Keep only available columns
        available_columns = [col for col in powerbi_columns if col in self.df.columns]
        final_df = self.df[available_columns].copy()
        
        # Clean data types for Power BI
        for col in final_df.columns:
            if final_df[col].dtype == 'object':
                final_df[col] = final_df[col].astype(str).str.strip()
                final_df[col] = final_df[col].replace('nan', '')
            elif final_df[col].dtype == 'float64':
                final_df[col] = final_df[col].round(3)
        
        # Sort by timestamp
        final_df = final_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Final dataset ready: {len(final_df)} rows, {len(final_df.columns)} columns")
        
        return final_df
    
    def generate_summary(self, final_df):
        print("\n" + "="*50)
        print("CLEAN DATASET SUMMARY")
        print("="*50)
        
        # Basic stats
        print(f"\nDataset Overview:")
        print(f"   Total Records: {len(final_df):,}")
        print(f"   Unique Users: {final_df['user_id'].nunique():,}")
        print(f"   Date Range: {final_df['date'].min()} to {final_df['date'].max()}")
        print(f"   Columns: {len(final_df.columns)}")
        
        # A/B Testing Summary
        v1_count = final_df['is_version_v1'].sum()
        v2_count = final_df['is_version_v2'].sum()
        print(f"\nA/B Testing:")
        print(f"   V1 Prompts: {v1_count:,}")
        print(f"   V2 Prompts: {v2_count:,}")
        print(f"   Winner: {final_df['ab_test_result'].iloc[0]}")
        
        # Performance Summary
        overall_success = final_df['is_successful'].mean()
        avg_rating = final_df['feedback_rating'].mean()
        print(f"\nPerformance:")
        print(f"   Success Rate: {overall_success:.1%}")
        print(f"   Average Rating: {avg_rating:.2f}/5")
        print(f"   High Ratings: {final_df['is_high_rating'].mean():.1%}")
        
        # User Insights
        power_users = final_df['is_power_user'].mean()
        premium_users = final_df['is_premium'].mean()
        print(f"\nUsers:")
        print(f"   Power Users: {power_users:.1%}")
        print(f"   Premium Users: {premium_users:.1%}")
        
        # Usage Patterns
        peak_hour = final_df.groupby('hour').size().idxmax()
        busiest_day = final_df.groupby('day_name').size().idxmax()
        print(f"\nUsage Patterns:")
        print(f"   Peak Hour: {peak_hour}:00")
        print(f"   Busiest Day: {busiest_day}")
        print(f"   Weekend Usage: {final_df['is_weekend'].mean():.1%}")
    
    def run_complete_cleaning(self):
        try:
            # Step 1: Load and clean
            self.load_and_clean_data() 
            # Step 2: Create time features
            self.create_time_features()
            # Step 3: Create text features
            self.create_text_features()
            # Step 4: Create business metrics
            self.create_business_metrics()
            # Step 5: Create user sessions
            self.create_user_sessions()
            # Step 6: Create A/B analysis
            self.create_simple_ab_analysis()
            # Step 7: Create user insights
            self.create_user_insights()
            # Step 8: Create final dataset
            final_dataset = self.create_final_dataset()
            # Step 9: Save the dataset
            output_file = "clean_powerbi_dataset.csv"
            final_dataset.to_csv(output_file, index=False)
            # Step 10: Generate summary
            self.generate_summary(final_dataset)  
            return final_dataset
            
        except Exception as e:
            print(f"Error in cleaning pipeline: {str(e)}")
            raise

# MAIN EXECUTION
if __name__ == "__main__":
    # Check if we have the basic requirements
    try:
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"Missing required library: {e}")
        print("Install with: pip install pandas numpy")
        exit()
    # Initialize and run cleaner
    cleaner = SimpleDatasetCleaner("enhanced_prompt_logs.csv")
    try:
        # Run complete cleaning
        clean_dataset = cleaner.run_complete_cleaning()
        
    except Exception as e:
        print(f"Error: {str(e)}")
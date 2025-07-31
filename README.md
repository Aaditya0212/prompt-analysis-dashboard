# Prompt Analysis & Optimization Dashboard

This project processes raw prompt log data into a clean, structured dataset ready for analysis in Power BI. It includes data cleaning, feature engineering, A/B test evaluation, and user behavior insights.

## Project Description

The goal of this project is to prepare a dataset that enables effective analysis of prompt interactions, user engagement, session patterns, and version-based performance. The final output is a CSV file optimized for use in Power BI dashboards.

## Features

- Data cleaning and missing value handling
- Time-based feature extraction (hour, day, weekday, business hours)
- Text analysis (word count, question detection, code/help/creative classification)
- Business metrics (success indicators, feedback quality, engagement score)
- Session identification and session-level insights
- A/B testing comparison between prompt versions
- User-level segmentation and insight generation
- Final dataset saved as a single CSV file with clean, formatted fields

## Input File

- `enhanced_prompt_logs.csv`: The raw dataset containing prompt logs.

## Output File

- `clean_powerbi_dataset.csv`: A cleaned dataset with enriched features, ready to upload to Power BI.

## How to Run the Script

1. Make sure Python is installed on your system.
2. Install required libraries if not already installed:
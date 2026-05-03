import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("dataset/cleaned_crimes.csv")
if not os.path.exists("eda"):
    os.makedirs("eda")

# Basic EDA: Crime Type Distribution
crime_counts = df['Primary Type'].value_counts()

crime_counts.plot(kind='bar', figsize=(12,6))
plt.title('Crime Distribution by Type')
plt.xlabel('Crime Type')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("eda/crime_distribution.png", bbox_inches='tight')
plt.close()

# Pie chart for crime distribution
def autopct_func(pct):
    return f'{pct:.1f}%' if pct > 5 else ''  # show only big slices

total = crime_counts.sum()
percentages = (crime_counts / total * 100).round(1)

legend_labels = [
    f"{label} ({pct}%)"
    for label, pct in zip(crime_counts.index, percentages)
]

ax = crime_counts.plot(
    kind='pie',
    figsize=(8,8),
    autopct=autopct_func,
    labels=None
)

plt.title('Crime Distribution by Type')
plt.ylabel('')

ax.legend(
    legend_labels,
    title="Crime Type",
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

plt.tight_layout()
plt.savefig("eda/crime_distribution_pie.png", bbox_inches='tight')
plt.close()

# Temporal Analysis: Crime Trends Over Time
hourly_counts = df['Date_hour'].value_counts().sort_index()

hourly_counts.plot(kind='bar', figsize=(12,5))
plt.title("Hourly Crime Distribution")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=0)
plt.savefig("eda/hourly_crime_distribution.png", bbox_inches='tight')
plt.close()

# Daily crime pattern (day of month)
daily_counts = df['Date_day'].value_counts().sort_index()

daily_counts.plot(kind='line', figsize=(12,5), marker='o')
plt.title("Daily Crime Pattern (Day of Month)")
plt.xlabel("Day")
plt.ylabel("Number of Crimes")
plt.savefig("eda/daily_crime_pattern.png", bbox_inches='tight')
plt.close()

# Monthly crime trend
monthly_counts = df['Date_month'].value_counts().sort_index()

monthly_counts.plot(kind='bar', figsize=(12,6))
plt.title("Monthly Crime Trend")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.xticks(rotation=0)
plt.savefig("eda/monthly_crime_trend.png", bbox_inches='tight')
plt.close()

# Seasonal analysis
season_counts = df['Season'].value_counts()

season_counts.plot(kind='bar', figsize=(12,6))
plt.title("Seasonal Crime Distribution")
plt.xlabel("Season")
plt.ylabel("Number of Crimes")
plt.savefig("eda/seasonal_crime_distribution.png", bbox_inches='tight')
plt.close()


# Weekly crime pattern
weekday_counts = df['Date_dayofweek'].value_counts().sort_index()

weekday_counts.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

weekday_counts.plot(kind='bar', figsize=(12,6))
plt.title("Weekly Crime Pattern")
plt.xlabel("Day of Week")
plt.ylabel("Number of Crimes")
plt.savefig("eda/weekly_crime_pattern.png", bbox_inches='tight')
plt.close()


# Nighttime crime analysis (8PM–11PM)
night_crimes = df[df['Date_hour'].between(20, 23)]

night_crimes['Primary Type'].value_counts().plot(
    kind='bar',
    figsize=(12,6)
)

plt.title("Crimes at Night (8PM–11PM)")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.savefig("eda/night_crimes.png", bbox_inches='tight')
plt.close()

# Arrest rate analysis
arrest_rate = df['Arrest'].value_counts(normalize=True) * 100
print(arrest_rate)

# Arrest distribution
df['Arrest'].value_counts().plot(
    kind='bar',
    figsize=(6,4)
)

plt.title("Arrest Distribution")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda/arrest_distribution.png", bbox_inches='tight')
plt.close()

# Arrest rate by crime type
arrest_by_crime = df.groupby('Primary Type')['Arrest'].mean().sort_values(ascending=False)

arrest_by_crime.plot(kind='bar', figsize=(12,5))

plt.title("Crimes by Arrest Rate")
plt.ylabel("Arrest Rate")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("eda/arrest_by_crime.png", bbox_inches='tight')
plt.close()

# Domestic crime analysis
df['Domestic'].value_counts(normalize=True) * 100

# Domestic vs Non-Domestic crime distribution
df['Domestic'].value_counts().plot(
    kind='bar',
    figsize=(6,4)
)

plt.title("Domestic vs Non-Domestic Crimes")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda/domestic_vs_non_domestic.png", bbox_inches='tight')
plt.close()


# Domestic crime rate by crime type
domestic_crime = df.groupby('Primary Type')['Domestic'].mean().sort_values(ascending=False)

domestic_crime.plot(kind='bar', figsize=(12,5))

plt.title("Crimes with Domestic Involvement Rate")
plt.ylabel("Domestic Rate")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("eda/domestic_crime_rate.png", bbox_inches='tight')
plt.close()

# Arrest rate for domestic vs non-domestic crimes
df.groupby('Domestic')['Arrest'].mean().plot(kind='bar', figsize=(6,4))

plt.title("Arrest Rate: Domestic vs Non-Domestic")
plt.ylabel("Arrest Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda/arrest_rate_domestic.png", bbox_inches='tight')
plt.close()

# Correlation between Arrest and Domestic
df['Arrest_num'] = df['Arrest'].astype(int)
df['Domestic_num'] = df['Domestic'].astype(int)

corr = df[['Arrest_num', 'Domestic_num']].corr()
corr.to_csv("eda/correlation_matrix.csv")
plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.savefig("eda/domestic_arrest_probability_correlation_heatmap.png", bbox_inches="tight")
plt.close()

# Arrest rate for domestic vs non-domestic crimes
domestic_arrest_rate = df.groupby('Domestic')['Arrest'].mean()
domestic_arrest_rate.to_csv("eda/domestic_arrest_rate.csv")

# Correlation between Arrest and Date features
corr = df[['Arrest_num', 'Date_hour', 'Date_dayofweek']].corr()

plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.savefig("eda/date_arrest_correlation_heatmap.png", bbox_inches="tight")
plt.close()

# Correlation between Arrest and Domestic (crosstab)
corr = pd.crosstab(df['Domestic'], df['Arrest'], normalize='index') * 100
plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.savefig("eda/domestic_arrest_correlation_heatmap.png", bbox_inches="tight")
plt.close()

# Crosstab for Domestic vs Arrest
ct = pd.crosstab(df['Domestic'], df['Arrest'])

ct.plot(
    kind='bar',
    stacked=True,
    figsize=(6,5)
)

plt.title("Domestic vs Arrest Distribution")
plt.xlabel("Domestic Case")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.legend(title="Arrest")
plt.tight_layout()
plt.savefig("eda/domestic_arrest_distribution.png", bbox_inches="tight")
plt.close()

# Crosstab for Domestic vs Arrest (probability)
ct = pd.crosstab(df['Domestic'], df['Arrest'], normalize='index') * 100

ct.plot(
    kind='bar',
    stacked=True,
    figsize=(6,5)
)

plt.title("Arrest Probability within Domestic vs Non-Domestic")
plt.xlabel("Domestic Case")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.legend(title="Arrest")
plt.tight_layout()
plt.savefig("eda/domestic_arrest_probability.png", bbox_inches="tight")
plt.close()

# Save EDA results to Excel
output_file = "eda/Crime_EDA_Report.xlsx"
writer = pd.ExcelWriter(output_file, engine='openpyxl')

# 1. Numeric summary
numeric_cols = ['Date_hour', 'Date_day', 'Date_month']
numeric_summary = df[numeric_cols].describe()
numeric_summary.to_excel(writer, sheet_name='Numeric Summary')

# 2. Categorical distributions
categorical_cols = ['Primary Type', 'Crime Group', 'Arrest', 'Domestic']
for col in categorical_cols:
    counts = df[col].value_counts()
    counts.to_frame(name='Count').to_excel(writer, sheet_name=f'{col} Distribution')

# 3. Arrest rate by crime type
arrest_rate_by_crime = df.groupby('Primary Type')['Arrest'].mean().sort_values(ascending=False)
arrest_rate_by_crime.to_frame(name='Arrest Rate').to_excel(writer, sheet_name='Arrest Rate by Crime')

# 4. Domestic involvement by crime type
domestic_rate_by_crime = df.groupby('Primary Type')['Domestic'].mean().sort_values(ascending=False)
domestic_rate_by_crime.to_frame(name='Domestic Rate').to_excel(writer, sheet_name='Domestic Rate by Crime')

# 5. Arrest rate by Domestic
arrest_by_domestic = df.groupby('Domestic')['Arrest'].mean()
arrest_by_domestic.to_frame(name='Arrest Rate').to_excel(writer, sheet_name='Arrest by Domestic')

# 6. Temporal patterns
temporal_cols = ['Date_hour', 'Date_dayofweek', 'Date_month', 'Season']
for col in temporal_cols:
    summary = df.groupby(col)['Primary Type'].count()
    summary.to_frame(name='Crime Count').to_excel(writer, sheet_name=f'Crime by {col}')

# 7. Correlation matrix
df['Arrest_num'] = df['Arrest'].astype(int)
df['Domestic_num'] = df['Domestic'].astype(int)
corr = df[['Arrest_num', 'Domestic_num', 'Date_hour', 'Date_dayofweek', 'Date_month']].corr()
corr.to_excel(writer, sheet_name='Correlation Matrix')
writer.close()
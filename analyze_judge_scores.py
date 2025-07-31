import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
# Visualization imports
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress convergence warnings for mixedlm
warnings.simplefilter('ignore', ConvergenceWarning)

def main():
    # Step 1: Load and preprocess data
    df = pd.read_csv('judge_scores_long.csv')
    df = df.dropna(subset=['score', 'answering_llm', 'difficulty', 'judge_llm', 'nickname', 'Talk'])
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df.dropna(subset=['score'])
    # Categorical variables
    df['answering_llm'] = df['answering_llm'].astype('category')
    df['difficulty'] = df['difficulty'].astype('category')
    df['Talk'] = df['Talk'].astype('category')
    df['judge_llm'] = df['judge_llm'].astype('category')
    df['nickname'] = df['nickname'].astype('category')

    print('Data loaded. N =', len(df))
    print('Columns:', df.columns.tolist())
    print(df.head())

    # Only use Audience and Speaker as judge_llm
    df = df[df['judge_llm'].isin(['Audience', 'Speaker'])]

    # Step 2: Fit linear mixed-effects model
    print('\nFitting mixed-effects model: Score ~ answering_llm + difficulty + Talk + (1|nickname)')
    md = smf.mixedlm("score ~ answering_llm + difficulty + Talk", df, groups=df["nickname"])
    mdf = md.fit()
    print(mdf.summary())

    # Step 3: Univariate associations
    print('\nUnivariate associations:')
    for var in ['answering_llm', 'difficulty', 'Talk', 'judge_llm']:
        print(f'\nUnivariate model: Score ~ {var}')
        model = smf.ols(f"score ~ C({var})", data=df).fit()
        print(model.summary())

    # Step 4: Test for interactions (Model:QuestionDifficulty)
    print('\nTesting interaction: answering_llm:difficulty')
    model_inter = smf.ols("score ~ answering_llm * difficulty", data=df).fit()
    print(model_inter.summary())


# Step 5: Visualization
    print("\nGenerating visualization plot...")
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    # Barplot: mean score by answering_llm, colored by difficulty, faceted by Talk
    g = sns.catplot(
        data=df,
        x="answering_llm",
        y="score",
        hue="difficulty",
        col="Talk",
        kind="bar",
        ci="sd",
        height=5,
        aspect=0.9,
        legend_out=True
    )
    g.set_axis_labels("Answering LLM", "Mean Score")
    g.set_titles("Talk: {col_name}")
    plt.subplots_adjust(top=0.85)
    plt.suptitle("Mean Score by Answering LLM, Difficulty, and Talk (Audience/Speaker Judges)")
    plt.savefig("judge_score_analysis_plot.png", bbox_inches="tight")
    plt.show()
    print("Plot saved as judge_score_analysis_plot.png")

if __name__ == "__main__":
    main()

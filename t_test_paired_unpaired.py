import pandas as pd
res = pd.read_csv('judge_scores_long.csv')
check_for = 'Audience'
res.loc[res['judge_llm'] == check_for,:]
res.loc[res['judge_llm'] == check_for,:]['nickname']
res.loc[res['judge_llm'] == check_for,:]['nickname'].value_counts()
import scipy
res2 = res.loc[res['judge_llm'] == check_for,:]
openai_scores = res2.loc[res2['answering_llm'] == 'openai',:].values
perplexity_scores = res2.loc[res2['answering_llm'] == 'perplexity',:].values

print(scipy.stats.ttest_rel(list(openai_scores[:,3]), list(perplexity_scores[:,3])))
print(scipy.stats.ttest_ind(list(openai_scores[:,3]), list(perplexity_scores[:,3])))

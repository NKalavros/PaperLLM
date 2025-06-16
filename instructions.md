# ISMB 2025 - {Session Name}

## Academic Talk Summarizer: User Guide

## Overview

We are organizing a crowdsourced effort to assess the capabilities of LLMs, specifically focused on academics. You will utilize a platform that benchmarks Large Language Models (LLMs) by having them answer questions about conference proceedings.

It collects quality ratings from both audience members and speakers to evaluate which LLMs provide the best responses to academic content. All information collected from you is anonymous with the exception of your nickname, which you provide.

We wanted to provide some guidelines on how we envision the platform to be used

---

## If You Are in the Audience

### Step 1: Ask Questions About a Talk (Audience Questions Tab)

1. **Select existing talk**: Choose from the talks
    - Note that you can ask while the talk is ongoing and your questions will get routed to the LLM after the talk has been completed and transcribed.

2. **Ask your question:**
   - Enter a specific question about the talk content.
   - The default prompt is: "Summarize the following papers key findings within 5 lines", but you can utilize any prompt you want
   - Be specific - good questions lead to better comparisons!
   - Try to stay in the scope of the talk - Ideally the question should be something the speaker mentioned explicitly, or at least touched upon.

3. **Set question difficulty:**
   - **Easy**: Basic comprehension questions
   - **Hard**: Complex analysis or synthesis questions

4. **Enter your nickname:**
   - It is crucial that yo use a consistent nickname to track your questions, you will need to remember them.
   - This helps you find your questions later

5. **Submit Question**
   - Answers typically generate in 30-60 seconds
   - Two LLMs (from OpenAI and Perplexity) will answer your question

### Step 2: Rate LLM Answers (LLM Answers Tab)

1. **Enter your nickname** (same one you used for questions)

2. **Set "Extra Questions"** (optional):
   - Enter 0 to see only your questions
   - Enter a number (e.g., 5) to also rate other people's questions

3. **Click "Load Answers"**

4. **For each question:**
   - Click "Show Details" to see both model responses
   - **Select Preferred Answer**: Choose Model 1 or Model 2
   - **Rate Quality**: Score each model 1-10
   - If you select a preferred model, ensure its quality score is higher than the other
   - Please try to submit ratings to all questions you asked. If that it too time consuming, you can go to the next step, but unrated questions will be nulled.

5. **Submit All Ratings** when complete

### Step 3: View Results (Leaderboard Tab)

- **Audience Leaderboard**: Shows average quality scores by difficulty level
- **Speaker Leaderboard**: Shows scores from speaker evaluations only
- Statistical significance (t-test, p-values) indicates if differences are meaningful

---

## If You Are a Speaker

As a speaker, you have access to a special interface to evaluate how well LLMs understood and can answer questions about your talk.

### Using the Speaker-Only Interface Tab

1. **Select your talk** from the dropdown menu
   - Only talks that have been uploaded will appear

2. **Choose number of questions** to review
   - Default is 5 questions
   - These are randomly selected from audience questions about your talk

3. **Click "Load Audience Questions & LLM Answers"**

4. **Rate each question's answers:**
   - Evaluate how accurately each model understood your talk
   - Consider technical accuracy and nuance
   - **Preferred Answer**: Which model better captured your points?
   - **Quality scores**: Rate each model 1-10

5. **Submit Ratings** when complete
   - Your ratings are specially marked as coming from the speaker
   - They contribute to the separate "Speaker Leaderboard"

### Why Speaker Ratings Matter

- You have unique insight into whether LLMs correctly understood your work
- Your ratings help identify which models best handle technical academic content
- The platform tracks speaker vs. audience ratings separately to compare perspectives
- These will be compared with LLM as a judge post-hoc to evaluate the accuracy of this approach in this setting

---

## Best Practices

### For Audience Members:
- Ask specific, focused questions rather than generic ones
- Be consistent with quality ratings across questions
- Consider rating some extra questions to help improve the dataset
- Use the same nickname consistently (or keep track of your nicknames)
- Do not use case-sensitive nicknames or extremely generic ones. Due to not collecting any eponymous data, name clashes can occur (e.g. 2 people using the nickname John)

### For Speakers:
- Focus on technical accuracy when rating responses
- Consider whether the LLM captured nuanced points from your talk
- Rate based on how well you'd want this summary to represent your work
- Your expert evaluation is valuable for improving AI understanding of academic content

---

## Understanding the Leaderboards

- **Mean ± SEM**: Average quality score ± standard error
- **Easy vs Hard**: Performance on different question difficulties  
- **N**: Number of ratings collected
- **p-value**: Statistical significance of differences between models
  - p < 0.05 suggests meaningful difference
  - Lower p-values indicate stronger evidence

The platform helps identify which LLMs best understand and communicate academic content, benefiting both researchers and the broader academic community.
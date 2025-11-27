# **Exploring Deep Learning to Develop Automatic Essay Scoring Models**

## **Purpose of this project**

Nowadays, students are studing on different online learning platforms. These platforms also provide assessments to show learners' progress. Multiple-choice questions are easy to be grade because there are correct answers. However, open-ended questions or essays are harder to grade. Peer review can be designed as an activity. When human graders are involved, they need to spend a large amout of time first reading the responses and then giving the score. Many education companies are trying to develop automatic essay scoring models, but encounter some problems in model development or implementation. Fortunately, we now have deep learning models and pre-trained large language models. This project will try to develop an automatic essay scoring model through deep learning.

## **Goal of this project**

The aim of this project is to use different deep learning models and a pre-trained large language model to develop an automatic essay scoring model. This is supervised multi-classification problem because the labels are provided and the labels are 1, 2, 3, 4, 5, 6. In the project, I will use a simple deep learning model (a multilayer perceptron), three sequential neural networks (bi-RNN, bi-LSTM, bi-GRU), and a pre-trained large language model (DeBERTa-V3-Base), and then compare their performance.

## **Dataset**

The data is from https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data. The competition has been closed. I selected the training data and just used the training data to develop different models. This dataset consists of 17307 observations and 3 variables (*essay_id*, *full_text*, and *score*). The variable of "full_text" is about students' essays and score is about their essay scores.

## **Exploratory data analysis**
I first checked whether there were missing values in the text and score columns. No missing values. Then, I used histogram to visualize the score variable. We could see there was huge imbalance of data in this dataset which could post a big challenge to the model performance. There are over 6000 essays with a score of 3, over 4000 essays with a score of 2, around 4000 essays with a score of 4. There are just over 1000 essays each for score 1 and 5. The highest score, 6, is lowest, with around 1000 essays. My hypothesis is that the model would achieve a better performance in scores like 2, 3, 4 but have a low performance in score 6. Or possibly, the model could not predict successfully in essays with score 6.

Here I would like to add two examples to show the essay text and its corresponding score.

**An essay with a score of 3**

*  Score 3 "I am a scientist at NASA that is discussing the "face" on mars. I will be explaining how the "face" is a land form. By sharing my information about this isue i will tell you just that.

   First off, how could it be a martions drawing. There is no plant life on mars as of rite now that we know of, which means so far as we know it is not possible for any type of life. That explains how it could not be made by martians. Also why and how would a martion build a face so big. It just does not make any since that a martian did this.

   Next, why it is a landform. There are many landforms that are weird here in America, and there is also landforms all around the whole Earth. Many of them look like something we can relate to like a snake a turtle a human... So if there are landforms on earth dont you think landforms are on mars to? Of course! why not? It's just unique that the landform on Mars looks like a human face. Also if there was martians and they were trying to get our attention dont you think we would have saw one by now?

   Finaly, why you should listen to me. You should listen to me because i am a member of NASA and i've been dealing with all of this stuff that were talking about and people who say martians did this have no relation with NASA and have never worked with anything to relate to this landform. One last thing is that everyone working at NASA says the same thing i say, that the "face" is just a landform.

   To sum all this up the "face" on mars is a landform but others would like to beleive it's a martian sculpture. Which every one that works at NASA says it's a landform and they are all the ones working on the planet and taking pictures."

**Another essay with a score of 6**

*   Score 6: Sometimes." (Plumer Paragraph 10) What the author is explaining is that during the presidential election, once you vote on your selection for the next president and you give your vote to the state electors you never know if they might change their mind or get scared and choose the incorrect candidate. For example, you choose on Barack Obama for president and you give your vote to the state electors and when it's time to vote... they decide to switch and choose HILARY CLINTON! Many of the citizens who voted for Barack Obama are now outraged by the thought of their state electors doing such a thing. If we had elections by popular vote we would be able to choose whom we specifically want for our President and there wouldn't be so much tension between people   

    Furthermore, the article "The indefensible electoral college: Why even the best-laid defenses of the system ae wrong" Bradford explains "Back in 1960, Segregationists in the Louisinna legislaure nearly succeeded in replacing the Democratic electors with new electors whoo would oppose John F. Kennedy." This quote from the article is saying that the electors could easily manipulate you and change their votes in order to get what they want, forgetting about all the other votes of the people back home waiting for the news that their selection has won the presidency.The elecoral college completely demolishes the purpose of the people's vote.

    Additionally, electoral colleges should be abolished because not everyone feels as strongly about it as they did hundreds of years ago when the process first came about. What had started out as a good idea has slowly turned into a unpredictable disaster. From time to time, People would be let down when they find out thatÂ  the candidate they had chosen didn't win the election, Why? because their state electors decided that it was okay for them to simply go against everyone else and be selfish by choosing their own candidate for presidency.

    Bradford proves this by explaining "...'faithless' electors have occasionally refused to vote for their party's candidate and cast a deciding vote for whomever they please..." (Plumer Paragraph 11) On multiple occasions voters have done exactly that, choosing someone completely different than whom they were supposed to. Many members of the party get angry with such childish behavior because it's selfish, uncalled for, and just disrespectful to go about ignoring the one major duty they had to cast a vote for their selected candidate. The article "In defense of the electoral college: Five reasons to keep our despised method of choosing the president" Richard A. Posner exclaims "The electoral college is widely regarded as a anachronism, a non-democratic method of selecting a president that ought to be [overruled] by declaring the candidate who recieves the most popular votes the winner." (Posner Paragraph 15) What the author is explaining is that the electoral college is an old custom and it's time that it was changed to something new like the election by popular vote.

    Time has changed, an so has the political veiws. The election by popular vote is a better opportunity because the state's people get to vote on exactly who they want without any major risks to deal with later on. Also, the election by popular vote is a simple and easier way of electing president.

    On the other hand, there are very few reasons that are pointing towards the electoral college being a good idea. For example, The electoral college has a even number of votes which make it easier to have a more predictable outcome of who might win the election. Although, not everyone might get the candidate that they had hoped for originally. The electoral college also comes along with the "Winner-take-all" method in which the awarding electoral votes induces the candidates running for the presidency. However, this is only based on the candidate that has the most popular votes. There are various reasons to consider the electoral college but many of them are followed by an overload of reasons NOT to keep the electoral college in use.

    Lastly, the election by popular vote should be used instead of the electoral college. The electoral college comes along with many complications and difficulties unlike the election by popular vote it has a simple and easier way of choosing who you want in the next presidency. Many people feel that you should change over to the election by popular vote to benefit all of the state's people so that they can have a more acurrate estimation of who theyÂ  might have as their new president.

    According to Bradford, the electoral college is "...Unfair, outdated, and irrational." (Plumer paragraph 14) It's about time we got rid of it and changed the way we elected our new president.

These essays are discussing different topics. The essay with a score of 6 obviously had a better argumentation, language organization and sophisticated thinking.

## **Feature Engineering**
I engineered six basic variables to represent the quality of essays: text_length, word_count, sentence_count, spelling_errors, readability_score (i.e., whether the text is easy to read and understand), and grade_level (i.e., which grade is the text suitable for). I then created a correlation analysis and remove readability_score and grade_level because of low correlation with the score.

## Model architecture
- I first used the four engineered variables to build a 3-layer multilayer perceptron model to predict the final score. I splitted the data 80% for training, 10% for validation, and 10% for testing. For this model, I chose ReLu activation and 0.3 for dropout. The first hidden layer had 64 neurons, the second hidden layer had 32 neurons, and the third hidden layer had 16 neurons. I selected cross entropy loss, adam optimizer and 50 epochs. Here I actually did some hyperparameter tuning, trying different learning rates and different epochs. I chose this model because I engineered four variables and attempted to use this model as the baseline model.

- Then, I built a sequential neural network pipeline (bi_RNN, bi_LSTM, bi_GRU). I chose these models because these essays are in long sequences and these models can better capture the temporal flow of information. I directly chose bidirectional because I think essay meaning is often influenced by both previous and upcoming context, and reading sequences in both forward and backward directions can help the model better understand relationships across the entire text. Before building models, I first created a word embedding with all the vocabularies in the text and gave them index. I used padding for those less than 300 words and unknown words. I implemented a unified recurrent model framework, RNNFamilyClassifier, which supports RNN, LSTM, and GRU architectures within a single pipeline. The model consists of an embedding layer, one recurrent layers (unidirectional or bidirectional), and a fully connected classification layer. Depending on the rnn_type parameter, the model dynamically initializes an RNN, LSTM, or GRU cell. Bidirectionality and depth are configurable, allowing the network to capture contextual information from both previous and upcoming tokens. The final prediction is obtained by concatenating the last forward and backward hidden states (if bidirectional) and passing them through a linear layer for classification.

- I finally used "microsoft/deberta-v3-base" as the primary model for fine-tuning, leveraging its pretrained language representations for sequence classification.

## **Model results and analysis**
- I fitted the models with some hyperparameter tuning, such as adding a new layer, changing the learning and increasing the number of epochs. Here is a summary of results:

| Model | Type | Overall ACC | Note |
|------|------|------------|--------|
| MLP | Feedforward | 0.46 | Did not predict score 1, 5, and 6 |
| Bi_RNN | Sequential | 0.46 | Performed worse in score 1; did not predict score 5 and 6 |
| Bi_LSTM | Sequential | 0.50 | Performed worse in score 1 and 5; did not predict score 6 |
| Bi_GRU | Sequential | 0.51 | Performed worse in score 1, 5, and 6 |
| DeBERTa-V3-Base | Transformer  |   0.65     |   Performed worse in score 5; did not predict score 6 |

- When I just used four engineered variables to predict the outcomes, the result was not good. The model almost could not predict any essays in score 1, 5, and 6. With sequential neural networks, the performance improved a little bit, but these models still struggled with essays with score 1, 5, and 6. The transformer DeBERTa-V3-Base improved the performance a lot. Now, it only struggled with essays with score 5, and 6. It actually had an average performance in score 5 but did not predict any essays in score 6. The main reason behind this is that there is a huge data imbalance. The number of essays in score 6 is pretty low. Even the transformer learned better and improve a lot, it still struggled with this.

## **Learning and takeaways**

- Transformers had a better performance than sequential neural network models and feedforward neural models considering it is better at capturing the contextual and temporal information.
- If there is a huge data imbalance, it is hard to build a model with a good performance because it is hard for the model to capture information from unrepresented categories.
- Although the final performance may not reach to a satisfactory score, it still shows a decent progress.

## **Ways to improve**
-  Using the transformer is the recommended choice. But some new features need to be engineered especially the unique features of score 6. For example, whether score 6 produced longer text; whether score 6 included more transition words to structure the text; whether score 6 presented a more persuasive style
-  Some data augmentation methods can be used. For example, consider using large language models to preduce responses in score 1, 5, and 6 to make the data more balanced. Still some unique features of essays of score 1, 5, and 6 should be distinguished first.

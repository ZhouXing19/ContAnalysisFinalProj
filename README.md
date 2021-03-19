# Can Machines Learn Finance?  --Textual Analysis of 10-K MD\&A

### Abstract

We conducted comprehensive content analysis on the Management Discussion and Analysis section of companies' annual report. We applied word count, topic modeling and word embedding methods to explore the industrial organization patterns, informational content related to stock returns and managerial visions and values. From word count, we find clear industrial organizational patterns and highly meaningful words associated with negative stock returns. We quantify the aggregate stock market sentiment using LM dictionary and find that the year 2020 has a significant impact on the real economy. From topic modeling, we were able to find latent topics both human-decipherable and undecipherable. The topic attention changes are used for stock return prediction. We observe the potential of the semantic features in stock return prediction. For the word embedding part, though the word2vec model did not provide significant structure of the embedding vector space, it gave intuitive results in mapping key words to the risk-uncertainty and profit-social responsibility dimensions.


### Author

Songrun He (hesongrun@uchicago.edu)
Chiayun Chang (cchiayun@uchicago.edu)
Zhou Xing (zhouxing@uchicago.edu)

### Acknowledgement
We want to express our sincere gratitude to Professor James Evans who provides us with continuous support and helpful comments. We also thank our two teaching assistants, Bhargav Srinivasa Desikan and Hyunku Kwon, for continuous technical support.

### Repository Structure

The `./codes` folder contains directories for the following topics, which correspond to sections in the report:

	- Corpus_cleanning
	- Corpus_construction
	- TopicModeling
	- WordEmbedding
	- Word_count

For the dataset, please check [here](https://www.dropbox.com/sh/0bgzsu0kzsjklw3/AAC3zs7XqP7lPR-WTiQ8s5nCa?dl=0).

### Notes for Computational Content Analysis Course

Techniques used in the projects come from:

- week 1 - count word frequency, and plot lexical dispersion and word cloud
- week 4 - classifying meanings and documents
- week 5 - implement LDA and clustering algorithms to capture topic dynamics
- week 7 - apply Word2Vec models and dimension projections


### References:

[1]  Lin  William  Cong,  Tengyuan  Liang,  and  Xiao  Zhang.   Textual  factors:   Ascalable,  interpretable,  and  data-driven  approach  to  analyzing  unstructuredinformation.Interpretable,  and  Data-driven  Approach  to  Analyzing  Unstruc-tured Information (September 1, 2019), 2019.

[2]  Fuwei Jiang, Joshua Lee, Xiumin Martin, and Guofu Zhou. Manager sentimentand stock returns.Journal of Financial Economics, 132(1):126–149, 2019.

[3]  Zheng Tracy Ke, Bryan T Kelly, and Dacheng Xiu.  Predicting returns withtext data.  Technical report, National Bureau of Economic Research, 2019.

[4]  Frank Knight.Risk, Uncertainty and Profit. Number 14 in Vernon Press Titlesin Economics. Vernon Art and Science Inc, July 2013.

[5]  Frank Hyneman Knight.Risk,  uncertainty  and  profit, volume 31.  HoughtonMifflin, 1921.

[6]  Tim Loughran and Bill McDonald.  When is a liability not a liability?  textualanalysis, dictionaries, and 10-ks.The Journal of finance, 66(1):35–65, 2011.

[7]  Paul C Tetlock.  Giving content to investor sentiment:  The role of media inthe stock market.The Journal of finance, 62(3):1139–1168, 2007

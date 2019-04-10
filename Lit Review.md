# Lit Review

## Data

- [Spanish](http://ps.clul.ul.pt/index.php?action=downloads)
    - CLUL (Ed.). 2014. P.S. Post Scriptum. Arquivo Digital de Escrita Quotidiana em Portugal e Espanha na Época Moderna. [last modified date]. URL: http://ps.clul.ul.pt.

---

## Papers

- Bollmann et al. (2019) - ["Few-Shot and Zero-Shot Learning for Historical Text Normalization"](https://arxiv.org/abs/1903.04870)
- Schneider et al. (2017) - ["Comparing Rule-based and SMT-based Spelling Normalisation for English Historical Texts"](https://www.aclweb.org/anthology/W17-0508)
- Bollmann et al. (2017) - ["Multi-task learning for historical text normalization: Size matters"](https://www.aclweb.org/anthology/W18-34#page=31)
- Eva Pettersson (2016) - ["Spelling Normalisation and Linguistic Analysis of Historical Text for Information Extraction"](https://www.diva-portal.org/smash/get/diva2:885117/FULLTEXT01.pdf)


### Bollmann et al (2019)
####Summary: 
The goal is to map "variant spellings in historical documents... to a common form, typically their modern equivalent." The researchers examined multitask learning for data in eight languages. Multitask learning is the process of training a model for multiple tasks concurrently. In addition to normalization, the researchers trained models for three "auxiliary" tasks: autoencoding, grapheme-to-phoneme mapping, and lemmatization. All three of these tasks relate to normalization in some way. Given a batch of 30 tokens for normalization, the model is exposed to 30 additional tokens, 10 for each auxiliary task. Multi-task learning involves some sharing of model weights between models for different tasks.
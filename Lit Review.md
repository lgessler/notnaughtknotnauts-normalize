# Lit Review

## The Task
Our project is focused on historical text normalization. Sometimes, this is also called historical text canonicalization, with the goal being to convert historical text into a standardized modern equivalent. This is a useful preprocessing step for research in the field of digital humanities

## Overview of the Literature
In many ways, the literature on this topic mirrors NLP research at large. Our selected references go back approximately ten years (though there are certainly older sources than that), and they exhibit a marked shift from heuristic, rule-based systems to complex statistical and neural models, though rule-based systems still perform well in certain cases. Most recently, the literature reflects a focus on statistical/neural machine translation, multi-task learning, attention, and deep learning.

---

## Selected Paper Summaries

#### "Few-Shot and Zero-Shot Learning for Historical Text Normalization"
##### Bollmann et al. (2019)

- Goal: map "variant spellings in historical documents... to a common form, typically their modern equivalent."
- Examined multitask learning for data in eight languages. Multitask learning is the process of training a model for multiple tasks concurrently.
- In addition to normalization, the researchers trained models for three "auxiliary" tasks: autoencoding, grapheme-to-phoneme mapping, and lemmatization, which relate to normalization in some way.
- Multitask Example: Given a batch of 30 tokens for normalization, the model is exposed to 30 additional tokens, 10 for each auxiliary task. Multi-task learning involves some sharing of model weights between models for different tasks.
- Few-shot and zero-shot learning in these experiments: "leverages existing
data from other languages, but assumes no labeled data for the target language."
- Key result: "[S]haring more components between main and auxiliary tasks is usually better."


#### "Evaluating historical text normalization systems: How well do they generalize?"
##### Robertson et al. (2018)

- Researchers compare two neural models against a baseline and find somewhat surprising negative results on downstream task of POS tagging
- Neural Normalization did not yield convincing performance gains
- They propose several guidelines / best practices for evaluating neural models in this context:
  - Report results on unseen tokens
  - Compare against a naïve baseline

#### "Comparing Rule-based and SMT-based Spelling Normalisation for English Historical Texts"
##### Schneider et al. (2017)

- Comparison of a dictionary-centric rule-based system and a statistical machine translation approach
- Uses ARCHER corpus (popular corpus for historical text normalization)
- Interesting observation that later texts are more standardized than earlier ones
  - 1600-1649: 315 per document; 1800-1849: 24
  - Large variance in number of non-standard spellings
- SMT system is trained on character sequences, not word sequences
- SMT system involves both a translation model and a language model
- Qualitative error analysis:
  - Some errors are actual mistakes; others are due to misannotation
  - apostrophe errors and word endings (e.g. flie vs. fly or captaine vs. captain) drag recall down
- Task is harder for later periods, where there's less spelling variation
- Conclusion: Rule-based system had peak accuracy, but SMT system generalizes better
- Future work: examining effects of normalization systems on downstream tasks like tagging or parsing

#### "Learning attention for historical text normalization by learning to pronounce"
##### Bollmann et al. (2017)

- Precursor to the Bollmann et al. (2019) paper
- Key insight is that the auxiliary task of "learning to pronounce" (i.e. grapheme-to-phoneme) can mimic the effects of attention, rendering attention redundant, which is useful

#### "Post Scriptum: Archivo Digital de Escritura Cotidiana"
##### Vaamonde et al. (2014)

- This is the paper describing the corpus we plan to use for our project
- Includes everyday documents in a less formal register than most published historical documents
- Contains original text and normalized text
- Includes both Spanish and Portuguese texts

#### "More Than Words: Using Token Context to Improve Canonicalization of Historical German"
##### Jurish (2010)

- Formal description of canonicalization task in terms of "conflation relations" and "conflators"
- Uses HMM to disambiguate between possible candidate tokens for normalization
- Explains stark precision-recall tradeoff in normalization tasks
- Shows that HMM can yield massive reduction in error compared to previous methods

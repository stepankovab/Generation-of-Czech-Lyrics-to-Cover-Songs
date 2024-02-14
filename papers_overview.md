# Overview of useful papers I've read

#### [A COMPUTATIONAL EVALUATION FRAMEWORK FOR SINGABLE LYRIC TRANSLATION](https://arxiv.org/abs/2308.13715)
- 2023
- navrhli metriky na porovnani dvou lyrics bez melody
    - count syllable distance
    - phoneme repetition correlation
    - music structure distance
    - semantic similarity
- syll. dist a sem. similarity jsou actually uzitecny, zbytek moc ne

#### [Automatic Song Translation for Tonal Languages](https://aclanthology.org/2022.findings-acl.60/)
- 2022
- pretraining transformer, adding length and tone constraints
- good Lyrics Generation section with methods

#### [“Poetic” Statistical Machine Translation: Rhyme and Meter](https://aclanthology.org/D10-1016/)
- 2010
- statistical, dealing with rhyme and meter
- more about exploring if it is managable to have constraints in statistical MT

#### [Dvojjazycne basnicky](https://www.vzjp.cz/verse.htm)
- dataaa

#### [Translation of musical theater into Spanish](https://repositori.upf.edu/bitstream/handle/10230/36040/Russell_2018.pdf?sequence=1&isAllowed=y)
- 2018
- Very linguistic
- Maybe some writing inspo

#### [Learned in Translation: Contextualized Word Vectors](https://proceedings.neurips.cc/paper_files/paper/2017/hash/20c86a628232a67e7bd46f76fba7ce12-Abstract.html)
- 2017
- mozna translation (spis na meaning to other language?)

#### [Singable translations of songs](https://www.tandfonline.com/doi/abs/10.1080/0907676X.2003.9961466)
- 2003
- Peter Low
- criteria of singability, linguistic, everybody cites this one







# TODO READ

## ASAP

#### [Songs Across Borders: Singable and Controllable Neural Lyric Translation](https://arxiv.org/abs/2305.16816)
- 2023

#### [Unsupervised Melody-Guided Lyrics Generation](https://arxiv.org/abs/2305.07760)
- 2023
- purely text based, without melody? 

#### [Zero-shot Sonnet Generation with Discourse-level Planning and Aesthetics Features](https://aclanthology.org/2022.naacl-main.262/)
- 2022

#### [Modeling the Rhythm from Lyrics for Melody Generation of Pop Song](https://arxiv.org/abs/2301.01361)
- 2022
- Mozna kdybych mela rytmus tak muzu z rytmu jednodusejc generovat dal

#### [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/)
- 2021
- continuous prompting
- !!!!!

#### [DeepRapper: Neural Rap Generation with Rhyme and Rhythm Modeling](https://arxiv.org/abs/2107.01875)
- 2021
- yo

#### [Controllable Generation from Pre-trained Language Models via Inverse Prompting](https://arxiv.org/abs/2103.10685)
- 2021
- $yo^2$

#### [Rigid Formats Controlled Text Generation](https://arxiv.org/abs/2004.08022)
- 2020
- Songnet - Transformer-based auto-regressive language model

#### [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension]()
- 2019
- Consider different transformers I guess?

#### [Integration of Dubbing Constraints into Machine Translation](https://aclanthology.org/W19-5210/)
- 2019
- dubbing, takze vlastne song lyrics bez ty melodie

#### [Neural Poetry Translation](https://aclanthology.org/N18-2011/)
- 2018

#### [Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation](https://aclanthology.org/N18-1119/)
- 2018
- omezuje output aby includoval urcity slova (anglicko-nemecky)

#### [Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search](https://aclanthology.org/P17-1141/)
- 2017
- incomporates knowledge into model without changing data or parameters

#### [Automatically Generating Rhythmic Verse with Neural Networks](https://aclanthology.org/P17-1016/)
- 2017
- The second approach considers poetry generation as a constraint satisfaction problem where a generative neural language model is tasked with learning a representation of content, and a discriminative weighted finite state machine constrains it on the basis of form

#### [DopeLearning: A Computational Approach to Rap Lyrics Generation](https://arxiv.org/abs/1505.04771)
- 2016
- Vybiraji z datasetu co maji, jenom kombinuji, netvori nove.
- Mohlo by byt zajimave co se tyce rhyme density
    - (neco jako kolik navzajem rymujicich slabik / pocet slabik)


## Theses

#### [Generátor anglické poezie s předtrénovanými jazykovými modely](https://dspace.cuni.cz/handle/20.500.11956/174292)
- 2022
- Bakalarka!!!!!

#### [Computational analysis and synthesis of song lyrics](https://dspace.cuni.cz/handle/20.500.11956/147665)
- 2021
- Diplomka

#### [Converting prose into poetry using neural networks](https://dspace.cuni.cz/handle/20.500.11956/148157)
- 2021
- Diplomka


## Maybe later

#### [A Comparison of Feature-Based and Neural Scansion of Poetry](https://aclanthology.org/R17-1003/)
- 2017
- Meter ruzny zpusoby ziskavani

#### [Automatic Analysis of Rhythmic Poetry with Applications to Generation and Translation](https://aclanthology.org/D10-1051/)
- 2010

#### [Controlling the Output Length of Neural Machine Translation](https://arxiv.org/abs/1910.10408)
- 2019
- Bez poetry, ale ma ruzne delky a ovlivnovani prekladu
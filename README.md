# Understanding Attention for Text Classification
This is the supplementary materials and Pytorch code for the paper Understanding Attention for Text Classification.
<p align="center">
 <img src="imgs/sst_attn_score_d10_tanh.png" width="700"/>
</p>
Attention has been proven successful in many natural language processing (NLP) tasks. Recently, many researchers started to investigate the interpretability of attention on NLP tasks.
Many existing approaches focused on examining whether the local attention weights could
reflect the importance of input representations. In this work, we present a study on understanding the internal mechanism of attention by looking into the gradient update process,
checking its behavior when approaching a local minimum during training. We propose to
analyze for each word token the following two
quantities: its polarity score and its attention
score, where the latter is a global assessment
on the token’s significance. We discuss conditions under which the attention mechanism
may become more (or less) interpretable, and
show how the interplay between the two quantities may impact the model performance

## Requirements
* Python version >= 3.6
* Pytorch version >= 1.1

## Data
* SST
* IMDB
* 20News I
* 20News II

## Citation
```
@inproceedings{sun-lu-2020-understanding,
    title = "Understanding Attention for Text Classification",
    author = "Sun, Xiaobing  and
      Lu, Wei",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.312",
    pages = "3418--3428",
    abstract = "Attention has been proven successful in many natural language processing (NLP) tasks. Recently, many researchers started to investigate the interpretability of attention on NLP tasks. Many existing approaches focused on examining whether the local attention weights could reflect the importance of input representations. In this work, we present a study on understanding the internal mechanism of attention by looking into the gradient update process, checking its behavior when approaching a local minimum during training. We propose to analyze for each word token the following two quantities: its polarity score and its attention score, where the latter is a global assessment on the token{'}s significance. We discuss conditions under which the attention mechanism may become more (or less) interpretable, and show how the interplay between the two quantities can contribute towards model performance.",
}
```


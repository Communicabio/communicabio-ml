
# communicabio-ml

Our solution for [Metachallenge hackathon](https://practicingfutures.org/meta).

This repo contains:

 - ``GPT-2`` deployment configs for ``Kubernetes`` (CPU + not working GPU) and ``Cloud Run``.
 - ``GPT-2`` are available in Russian and English
 - ``BERT`` for classification training scripts

Check [communicabio-hints](https://github.com/Communicabio/communicabio-hints) for ``BERT`` deployment configs

## Training

We tried to create two models:
1. For toxicity detection/measurement
2. For positivity detection

For deployment purposes we were unable to fine-tune ``BERT``*, thus, we added a small 1 linear layer network to ``BERT`` and trained it, instead of the whole model.

*To minimize server cold start time we aimed to unify ``BERT`` servers. Due to 2``GB`` ``Cloud Run`` restrictions, it`s impossible to store different BERT models in one image.  

## Datasets

### Sentiment

http://www.dialog-21.ru/evaluation/2016/sentiment/

https://gitlab.com/kensand/rusentiment

http://study.mokoron.com/

https://github.com/oldaandozerskaya/auto_reviews

### Toxic

http://tpc.at.ispras.ru/prakticheskoe-zadanie-2015/

https://www.kaggle.com/blackmoon/russian-language-toxic-comments/data

http://files.deeppavlov.ai/models/obscenity_classifier/ru_obscenity_dataset.zip

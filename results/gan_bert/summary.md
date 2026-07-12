# GAN-BERT experiment results

## Accuracy

| dataset   |   SPECTER2 |   SciBERT |   XLNet |   specter2_seed123 |
|:----------|-----------:|----------:|--------:|-------------------:|
| 3C        |     0.608  |    0.608  |  0.5893 |           nan      |
| ACL-ARC   |     0.7914 |    0.7914 |  0.7482 |           nan      |
| SciCite   |     0.5368 |    0.7614 |  0.6244 |             0.5357 |

## F1-macro

| dataset   |   SPECTER2 |   SciBERT |   XLNet |   specter2_seed123 |
|:----------|-----------:|----------:|--------:|-------------------:|
| 3C        |     0.4232 |    0.4139 |  0.3813 |           nan      |
| ACL-ARC   |     0.7148 |    0.6884 |  0.637  |           nan      |
| SciCite   |     0.2349 |    0.5444 |  0.3966 |             0.2326 |

## Full detail

| dataset   | model            |   accuracy |   f1_macro |   best_epoch |   wall_clock_seconds | status   |
|:----------|:-----------------|-----------:|-----------:|-------------:|---------------------:|:---------|
| 3C        | SciBERT          |     0.608  |     0.4139 |           24 |                 96.8 | ok       |
| 3C        | SPECTER2         |     0.608  |     0.4232 |           17 |                205.7 | ok       |
| 3C        | XLNet            |     0.5893 |     0.3813 |           25 |                236.3 | ok       |
| ACL-ARC   | SciBERT          |     0.7914 |     0.6884 |           22 |                146.3 | ok       |
| ACL-ARC   | SPECTER2         |     0.7914 |     0.7148 |           23 |                 70.7 | ok       |
| ACL-ARC   | XLNet            |     0.7482 |     0.637  |           16 |                103.6 | ok       |
| SciCite   | SciBERT          |     0.7614 |     0.5444 |           20 |                430.9 | ok       |
| SciCite   | SPECTER2         |     0.5368 |     0.2349 |            2 |                424.3 | ok       |
| SciCite   | specter2_seed123 |     0.5357 |     0.2326 |            2 |                420.8 | ok       |
| SciCite   | XLNet            |     0.6244 |     0.3966 |           20 |                858.9 | ok       |

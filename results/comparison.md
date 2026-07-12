# Baseline vs. GAN-BERT (vanilla) vs. GAN-BERT (improved)

## F1-macro

|                         |   No-GAN baseline |   GAN-BERT (vanilla) |   GAN-BERT (improved) |
|:------------------------|------------------:|---------------------:|----------------------:|
| ('3C', 'SPECTER2')      |            0.3967 |               0.4232 |                0.4297 |
| ('3C', 'SciBERT')       |            0.4001 |               0.4139 |                0.4079 |
| ('3C', 'XLNet')         |            0.3605 |               0.3813 |                0.3848 |
| ('ACL-ARC', 'SPECTER2') |            0.6677 |               0.7148 |                0.6672 |
| ('ACL-ARC', 'SciBERT')  |            0.6852 |               0.6884 |                0.7292 |
| ('ACL-ARC', 'XLNet')    |            0.6915 |               0.637  |                0.6625 |
| ('SciCite', 'SPECTER2') |            0.8392 |               0.2349 |                0.8393 |
| ('SciCite', 'SciBERT')  |            0.8372 |               0.5444 |                0.8381 |
| ('SciCite', 'XLNet')    |            0.8139 |               0.3966 |                0.8108 |

## Accuracy

|                         |   No-GAN baseline |   GAN-BERT (vanilla) |   GAN-BERT (improved) |
|:------------------------|------------------:|---------------------:|----------------------:|
| ('3C', 'SPECTER2')      |            0.624  |               0.608  |                0.6293 |
| ('3C', 'SciBERT')       |            0.6293 |               0.608  |                0.616  |
| ('3C', 'XLNet')         |            0.584  |               0.5893 |                0.6293 |
| ('ACL-ARC', 'SPECTER2') |            0.7842 |               0.7914 |                0.7914 |
| ('ACL-ARC', 'SciBERT')  |            0.7842 |               0.7914 |                0.8058 |
| ('ACL-ARC', 'XLNet')    |            0.7698 |               0.7482 |                0.7626 |
| ('SciCite', 'SPECTER2') |            0.8501 |               0.5368 |                0.8517 |
| ('SciCite', 'SciBERT')  |            0.849  |               0.7614 |                0.8533 |
| ('SciCite', 'XLNet')    |            0.8297 |               0.6244 |                0.8313 |

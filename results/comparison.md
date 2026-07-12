# GAN-BERT citation-intent experiments: full comparison

Dimensions varied: (1) partial fine-tuning (last 2 transformer layers + pooler) vs. full fine-tuning; (2) no-GAN baseline vs. vanilla GAN-BERT vs. improved GAN-BERT (label smoothing + Pi-model consistency regularization, and a corrected learning rate for SciCite); (3) in-domain vs. cross-dataset unlabeled data for the GAN discriminator's real/fake stream.

**Best overall result:** SciBERT / SciCite / GAN-BERT vanilla (full) -- F1-macro=0.8814, accuracy=0.8909

## F1-macro

|                         |   No-GAN baseline (partial) |   No-GAN baseline (full) |   GAN-BERT vanilla (partial) |   GAN-BERT vanilla (full) |   GAN-BERT improved (partial) |   GAN-BERT improved (full) |   GAN-BERT improved (partial, cross-dataset unlabeled) |
|:------------------------|----------------------------:|-------------------------:|-----------------------------:|--------------------------:|------------------------------:|---------------------------:|-------------------------------------------------------:|
| ('3C', 'SPECTER2')      |                      0.3967 |                   0.4012 |                       0.4232 |                    0.4027 |                        0.4297 |                     0.3824 |                                                 0.3953 |
| ('3C', 'SciBERT')       |                      0.4001 |                   0.4446 |                       0.4139 |                    0.3644 |                        0.4079 |                     0.4108 |                                                 0.4022 |
| ('3C', 'XLNet')         |                      0.3605 |                   0.416  |                       0.3813 |                    0.3552 |                        0.3848 |                     0.3454 |                                                 0.3637 |
| ('ACL-ARC', 'SPECTER2') |                      0.6677 |                   0.7422 |                       0.7148 |                    0.6968 |                        0.6672 |                     0.7361 |                                                 0.6921 |
| ('ACL-ARC', 'SciBERT')  |                      0.6852 |                   0.7826 |                       0.6884 |                    0.7266 |                        0.7292 |                     0.6364 |                                                 0.6999 |
| ('ACL-ARC', 'XLNet')    |                      0.6915 |                   0.6456 |                       0.637  |                    0.7362 |                        0.6625 |                     0.7013 |                                                 0.6122 |
| ('SciCite', 'SPECTER2') |                      0.8392 |                   0.8419 |                       0.2349 |                    0.8667 |                        0.8393 |                     0.8678 |                                                 0.84   |
| ('SciCite', 'SciBERT')  |                      0.8372 |                   0.8213 |                       0.5444 |                    0.8814 |                        0.8381 |                     0.8801 |                                                 0.833  |
| ('SciCite', 'XLNet')    |                      0.8139 |                   0.8313 |                       0.3966 |                    0.8241 |                        0.8108 |                     0.812  |                                                 0.8323 |

## Accuracy

|                         |   No-GAN baseline (partial) |   No-GAN baseline (full) |   GAN-BERT vanilla (partial) |   GAN-BERT vanilla (full) |   GAN-BERT improved (partial) |   GAN-BERT improved (full) |   GAN-BERT improved (partial, cross-dataset unlabeled) |
|:------------------------|----------------------------:|-------------------------:|-----------------------------:|--------------------------:|------------------------------:|---------------------------:|-------------------------------------------------------:|
| ('3C', 'SPECTER2')      |                      0.624  |                   0.6107 |                       0.608  |                    0.576  |                        0.6293 |                     0.5787 |                                                 0.6293 |
| ('3C', 'SciBERT')       |                      0.6293 |                   0.6693 |                       0.608  |                    0.5947 |                        0.616  |                     0.6267 |                                                 0.6107 |
| ('3C', 'XLNet')         |                      0.584  |                   0.6053 |                       0.5893 |                    0.5387 |                        0.6293 |                     0.5333 |                                                 0.584  |
| ('ACL-ARC', 'SPECTER2') |                      0.7842 |                   0.8129 |                       0.7914 |                    0.7914 |                        0.7914 |                     0.8129 |                                                 0.7842 |
| ('ACL-ARC', 'SciBERT')  |                      0.7842 |                   0.8058 |                       0.7914 |                    0.8201 |                        0.8058 |                     0.777  |                                                 0.777  |
| ('ACL-ARC', 'XLNet')    |                      0.7698 |                   0.7626 |                       0.7482 |                    0.8129 |                        0.7626 |                     0.7986 |                                                 0.7554 |
| ('SciCite', 'SPECTER2') |                      0.8501 |                   0.8544 |                       0.5368 |                    0.8802 |                        0.8517 |                     0.8802 |                                                 0.8538 |
| ('SciCite', 'SciBERT')  |                      0.849  |                   0.8366 |                       0.7614 |                    0.8909 |                        0.8533 |                     0.8904 |                                                 0.8469 |
| ('SciCite', 'XLNet')    |                      0.8297 |                   0.8469 |                       0.6244 |                    0.849  |                        0.8313 |                     0.8415 |                                                 0.8479 |

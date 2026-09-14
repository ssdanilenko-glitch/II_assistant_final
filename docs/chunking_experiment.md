# Chunking Experiment Results

| Стратегия | Hit Rate@5 | MRR@10 | Recall@10 | Ср. длина chunk | Скорость retrieval (мс) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| docs_fixed | 1.000 | 1.000 | 1.000 | 1300 | 316.7 |
| docs_recursive | 1.000 | 1.000 | 1.000 | 1330 | 268.5 |
| docs_semantic | 1.000 | 1.000 | 1.000 | 1400 | 251.1 |
| **docs_fixed (best baseline)** | 1.000 | 1.000 | 1.000 | 1300 | 246.2 |
| **docs_fixed + reranker** | 1.000 | 1.000 | 1.000 | 1300 | 23155.3 |

### Вывод
Выбираю стратегию **docs_fixed** с конфигом **chunk_size=512, overlap=64, top-K=20**, потому что она дает наилучший баланс между качеством (Hit Rate@5 = 1.000 с reranker'ом) и скоростью. Добавление cross-encoder `bge-reranker-v2-m3` увеличило Hit Rate@5 на 0.000 относительно базовой версии, что критично для многошаговых вопросов из golden dataset.
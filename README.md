# NLP-Helper
set of utility tools for NLP tasks.

## N-gram Masked Language Modeling Collator
    A N-gram MLM collator compatible with a torch DataLoader
```python
from utils.mlm_n_gram_collator import NGramDataCollatorForLanguageModeling
from torch.utils.data import DataLoader

collator = NGramDataCollatorForLanguageModeling(tokenizer=tokenizer, n_gram=3, mlm_probality=.15)
dataloader = DataLoader(dataset, collate_fn=collator)
```
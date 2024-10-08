import sys

sys.path.append('../')

import os
import traceback

from tqdm import tqdm
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline

from myconfig import cached_dir
import pandas as pd
df = pd.read_csv('sinomed_chinese_textual_corpus_for_translation.csv')

# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
# model = model.to('cuda')
# for i, row in tqdm(df.iterrows()):
#     id, title, author_name_with_aff_order, aff_arr, journal_title, pub_year = row
#     tokenized_text = tokenizer.prepare_seq2seq_batch([title], return_tensors='pt')
#     translation = model.generate(**tokenized_text)
#     translated_text = tokenizer.batch_decode(translation, skip_special_tokens=False)[0]
#     print(translated_text)

mode_name = 'liam168/trans-opus-mt-zh-en'
# mode_name = 'Helsinki-NLP/opus-mt-en-zh'
model = AutoModelWithLMHead.from_pretrained(mode_name)
tokenizer = AutoTokenizer.from_pretrained(mode_name)
translation = pipeline("translation_zh_to_en", model=model, tokenizer=tokenizer, device=0)

num = len(df)
starts = list(range(0, num, 150))
ends = starts[1:] + [num]

rg = list(zip(starts, ends))
# rg = list(zip(starts, ends))[::-1]
for (s, e) in tqdm(rg, total=len(starts)):
    print(s, e)
    try:
        sub_df = df[s:e]
        if len(sub_df) == 0:
            continue
        content_list = sub_df['content'].values.tolist()
        translated_text = translation(content_list, max_length=70)
        translated_text = [(n.get('translation_text') if n.get('translation_text') else '').replace('\n', ' ') for n in
                           translated_text]
        # print(content_list, translated_text)
        assert len(translated_text) == len(sub_df)
        del sub_df['content']
        sub_df['translated_content'] = translated_text
        sub_df.to_csv(os.path.join(cached_dir, 'xmol-pubscholar-sciencechina-translated-chinese-textual-corpus.csv'), sep=',', mode='a', header=False)
    except Exception as e:
        traceback.print_exc()

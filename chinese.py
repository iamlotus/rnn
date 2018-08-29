import collections

a="这是中文"
print(a)


"""
Process txt file specified by `file_path`
:param file_path: a txt file, echo poem in one line
:return: (poems_vector,word_idx_map,words), where poems_vector is list<list<wordIdx>>, word_idx_map is map<chinese word, wordIdx>, words is list of all words
"""
# 诗集
poems=[]
error_line=0
total_line=0
with open('data/demo_poems.txt','r',encoding='utf') as f:
    for line in f.readlines():
        total_line+=1

        if not ':' in line:
            error_line+=1
            continue
        try:
            title , content=line.strip().split(':')
            content=content.replace(' ','')
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content :
                error_line+=1
                continue

            content= '%s%s%s'%('S',content,'E')
            poems.append(content)
        except ValueError as e:
            error_line += 1
            pass



all_words= [word for poem in poems for word in poem]
# print("%d words totally"%len(all_words))

counter=collections.Counter(all_words)
counter_pairs= sorted(counter.items(),key=lambda x:x[1],reverse=True)
words, freqs = zip(*counter_pairs)
# padding with blank
words = words +(' ',)
word_idx_map=dict(zip(words,range(len(words))))
poems_vector=[list(map(lambda word:word_idx_map[word],poem)) for poem in poems]

print(u"共找到%d首诗, 有%d首无法处理, 一共处理%d首，共计%d个词, %d个诗向量" % (total_line, error_line, len(poems),len(words),len(poems_vector)))
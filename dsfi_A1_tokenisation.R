libraries <- 'libraries.R'
source(libraries)
setpwd()
load(file='./sonaTibble.RData')

# Tokenisation
sona_sentences <- sona %>% unnest_tokens(sentence, speech, token = 'sentences') %>% mutate(sentence_id=row_number())
sona_sentences <- sona_sentences %>% 
  left_join(sona_sentences %>% unnest_tokens(word, sentence, token = 'words') %>% group_by(sentence_id) %>% count() %>% ungroup()) %>% 
  mutate(word_count=n) %>% select(-n) %>%  # Add word count
  filter(word_count>2) # remove sentences of two words or less
sona_words <- sona_sentences %>% unnest_tokens(word, sentence, token = 'words')

dict <- sona_words %>%
  group_by(word) %>%
  count(sort=T) %>%
  ungroup()

sentence_dict <- sona_words %>%
  inner_join(dict) %>%
  group_by(sentence_id,word) %>%
  count() %>%  
  group_by(sentence_id) %>%
  mutate(total = sum(n)) %>%
  ungroup() %>% 
  left_join(sona_words %>% group_by(word) %>% 
  summarize(sentence_with_word = n()) %>% 
  ungroup())

sentence_count <- length(unique(sentence_dict$sentence_id))

bag_of_words <- sentence_dict %>% 
  select(sentence_id, word, n) %>% 
  pivot_wider(names_from = word, values_from = n, values_fill = 0) %>%
  left_join(sona_sentences %>% select(sentence_id, president_speaker), by = 'sentence_id') %>%
  select(sentence_id, president_speaker, everything())

sentence_dict <- sentence_dict %>% bind_tf_idf(word, sentence_id, n) 

idf_bag <- sentence_dict %>% 
  select(sentence_id, word, tf_idf) %>%  
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) %>%  
  left_join(sona_sentences %>% select(sentence_id,president_speaker), by = c('sentence_id')) %>%
  select(sentence_id, president_speaker, everything())


save(file='sonaTokens.RData', list=c('idf_bag', 'bag_of_words', 'sentence_count','sentence_dict', 'dict', 'sona_sentences'))

sentence_lengths <- sona_sentences %>% group_by(president_speaker, year) %>% 
  mutate(mean_count = mean(word_count)) %>%
  select(president_speaker, year, mean_count) %>% ungroup()

plot(sentence_lengths$year, sentence_lengths$mean_count, 
     col=as.factor(sentence_lengths$president_speaker), pch =19, cex=sentence_lengths$mean_count/10,
     ylab='Word Count Per Sentence', xlab='Year', main='Average Sentence Length by President')
legend('topright', legend=unique(sentence_lengths$president_speaker),
       col=as.factor(unique(sentence_lengths$president_speaker)), 
       pch =19, cex=1.2, bty='n')

name_counts <- sentence_dict %>% 
  filter(word %in% c('mandela','zuma','klerk','motlanthe', 'mbeki','ramaphosa')) %>% 
  #'nelson','jacob','thabo',,'cyril'
  inner_join(sona_sentences) %>% 
  group_by(president_speaker, word) %>% count() %>% ungroup() %>% 
  select(president_speaker, word, n) %>%
  pivot_wider(names_from = word, values_from = n, values_fill = 0) 

name_mat <- as.matrix(name_counts[,-1]); row.names(name_mat) <- name_counts$president_speaker;
barplot(t(name_mat), beside=T,legend.text = colnames(name_mat),
        args.legend = list(x='top', bty='n'), main='Frequency of President Names in Speeches') 




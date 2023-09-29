### Bag of Words Models
libraries <- 'libraries.R'
source(libraries)
setpwd()
load(file='./sonaTibble.RData')

# Tokenisation
sona_sentences <- sona %>% unnest_tokens(sentence, speech, token = 'sentences') %>% mutate(sentId=row_number())
sona_words <- sona_sentences %>% unnest_tokens(word, sentence, token = 'words')

word_bag <- sona_words %>%
  group_by(word) %>%
  count(sort=T) %>%
  ungroup()

sentences_tdf <- sona_words %>%
  inner_join(word_bag) %>%
  group_by(sentId,word) %>%
  count() %>%  
  group_by(sentId) %>%
  mutate(total = sum(n)) %>%
  ungroup()

bag_of_words <- sentences_tdf %>% 
  select(sentId, word, n) %>% 
  pivot_wider(names_from = word, values_from = n, values_fill = 0) %>%
  left_join(sona_sentences %>% select(sentId, filename)) %>%
  select(sentId, filename, everything())


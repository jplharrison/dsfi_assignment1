libraries <- 'libraries.R'
source(libraries)
setpwd()

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

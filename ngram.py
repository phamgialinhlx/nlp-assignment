from argparse import ArgumentParser
from collections import Counter
import spacy

nlp = spacy.load('en_core_web_sm')

corpus = (
    "Gus Proto is a Python developer currently"
    " working for a London-based Fintech company. He is"
    " interested in learning Natural Language Processing."
    " There is a developer conference happening on 21 July"
    ' 2019 in London. It is titled "Applications of Natural'
    ' Language Processing". There is a helpline number'
    " available at +44-1234567891. Gus is helping organize it."
    " He keeps organizing local Python meetups and several"
    " internal talks at his workplace. Gus is also presenting"
    ' a talk. The talk will introduce the reader about "Use'
    ' cases of Natural Language Processing in Fintech".'
    " Apart from his work, he is very passionate about music."
    " Gus is learning to play the Piano. He has enrolled"
    " himself in the weekend batch of Great Piano Academy."
    " Great Piano Academy is situated in Mayfair or the City"
    " of London and has world-class piano instructors."
)
corpus = (
    "The boy likes books. The"
    " books are bank"
    " books. Some books are on the table. Some"
    " boys books the table. The book has many"
    " likes. Many banks like books."
)
complete_doc = nlp(corpus)

class NGramModel:
    def __init__(self, sentences, n):
        self.build_ngrams(sentences, n)
        self.n = n

    def build_ngrams(self, sentences, n):
        self.v = len(set([token.lemma_ for token in complete_doc if not token.is_punct]))
        batches = []
        for sentence in sentences:
            tokenized_sentence = nlp(sentence)
            words = [token.lemma_ for token in tokenized_sentence if not token.is_punct]
            words = ['<s>'] + words + ['</s>']
            for i in range(len(words)-n+1):
                batches.append(words[i:i+n])
        self.ngrams = {}
        for batch in batches:
            key = ' '.join(batch[:-1])
            if key not in self.ngrams.keys():
                self.ngrams[key] = []
            self.ngrams[key].append(batch[-1])        

    def __str__(self):
        return str(self.ngrams)
    
    def mle(self, word, context):
        if context not in self.ngrams.keys():
            return 0
        count = Counter(self.ngrams[context])
        return count[word] / sum(count.values())
    
    def mle_smoothing(self, word, context):
        if context not in self.ngrams.keys():
            return 1 / self.v
        count = Counter(self.ngrams[context])
        return (count[word] + 1) / (sum(count.values()) + self.v)
    
    def generate(self, context='<s>', length=10):
        if context == '<s>':
            sentence = [context]
        else:
            tokenized_context = nlp(context)
            sentence = [token.lemma_ for token in tokenized_context if not token.is_punct]
        
        context = ' '.join(sentence[-self.n+1:])
        for i in range(length):
            word = max(Counter(self.ngrams[context]))
            if word == '</s>':
                break
            sentence.append(word)
            context = ' '.join(sentence[-self.n+1:])
        if sentence[0] == '<s>':
            sentence = sentence[1:]
        return ' '.join(sentence)

def main(args):
    n = args.n
    sentences = [sent.text for sent in complete_doc.sents]
    ngrams = NGramModel(sentences, n)
    print(ngrams.ngrams)
    print(ngrams.mle("book", "the"))
    print(ngrams.mle("book", "as"))
    print(ngrams.mle_smoothing("book", "the"))
    print(ngrams.mle_smoothing("book", "as"))
    print(ngrams.v)
    print(ngrams.generate(context="The books are", length=10))

if __name__ == '__main__':
    # create argument parser object
    argumentParser = ArgumentParser(description='ngram model')
    argumentParser.add_argument('--n', type=int, default=2)
    args = argumentParser.parse_args()
    main(args)

from argparse import ArgumentParser
from collections import Counter
import spacy

nlp = spacy.load('en_core_web_sm')

# corpus = (
#     "Gus Proto is a Python developer currently"
#     " working for a London-based Fintech company. He is"
#     " interested in learning Natural Language Processing."
#     " There is a developer conference happening on 21 July"
#     ' 2019 in London. It is titled "Applications of Natural'
#     ' Language Processing". There is a helpline number'
#     " available at +44-1234567891. Gus is helping organize it."
#     " He keeps organizing local Python meetups and several"
#     " internal talks at his workplace. Gus is also presenting"
#     ' a talk. The talk will introduce the reader about "Use'
#     ' cases of Natural Language Processing in Fintech".'
#     " Apart from his work, he is very passionate about music."
#     " Gus is learning to play the Piano. He has enrolled"
#     " himself in the weekend batch of Great Piano Academy."
#     " Great Piano Academy is situated in Mayfair or the City"
#     " of London and has world-class piano instructors."
# )
corpus = (
    "The boy likes books. The"
    " books are bank"
    " books. Some books are on the table. Some"
    " boys books the table. The book has many"
    " likes. Many banks like books."
)
complete_doc = nlp(corpus)

class NGramModel:
    def __init__(self, sentences, n, backoff=False, interpolation=False):
        assert not (backoff == True and interpolation == True), "Can't have both backoff and interpolation"
        self.backoff = backoff
        self.interpolation = interpolation
        self.full_context = backoff or interpolation
        if self.full_context:
            self.n = 3
        else:
            self.n = n
        self.build_ngrams(sentences, n)

    def build_ngrams(self, sentences, n):
        self.v = set([token.lemma_ for token in complete_doc if not token.is_punct])
        batches = []
        for sentence in sentences:
            tokenized_sentence = nlp(sentence)
            words = [token.lemma_ for token in tokenized_sentence if not token.is_punct]
            words = ['<s>'] + words + ['</s>']
            if self.full_context:
                for i in range(1, 4):
                    for j in range(len(words)-i+1):
                        batches.append(words[j:j+i])
            else:
                for i in range(len(words)-n+1):
                    batches.append(words[i:i+n])
            
        self.ngrams = {}
        for batch in batches:
            key = ' '.join(batch[:-1])
            if key not in self.ngrams.keys():
                self.ngrams[key] = {}
            self.ngrams[key].update({batch[-1]: self.ngrams[key].get(batch[-1], 0) + 1})        

    def __str__(self):
        return str(self.ngrams)

    def prob(self, word, context, laplace:float=1):
        """
        laplace=0: no smoothing
        """
        if context not in self.ngrams.keys():
            return 1 * laplace / len(self.v)
        count = Counter(self.ngrams[context])
        if not self.backoff and self.interpolation:
            return (self.ngrams[context][word] + 1 * laplace) / (sum(count.values()) + len(self.v) * laplace)
        elif self.backoff:
            divided_context = [token.lemma_ for token in nlp(context) if not token.is_punct]
            backoff_context = [context, divided_context[0], ""]
            for bc in backoff_context:
                if bc in self.ngrams.keys():
                    return (self.ngrams[context][word] + 1 * laplace) / (sum(count.values()) + len(self.v) * laplace)
        elif self.interpolation:
            divided_context = [token.lemma_ for token in nlp(context) if not token.is_punct]
            interpolation_context = [context, divided_context[0], ""]
            p = 0.0
            for i in range(len(interpolation_context)):
                p += (self.ngrams[interpolation_context[i]][word] + 1 * laplace) / (sum(count.values()) + len(self.v) * laplace)
            return p
    def generate(self, context='<s>', length=10):
        if context == '<s>':
            sentence = [context]
        else:
            tokenized_context = nlp(context)
            sentence = [token.lemma_ for token in tokenized_context if not token.is_punct]

        if self.n == 1:
            return context + ' ' + ' '.join([max(Counter(self.ngrams[""]))for i in range(length)])
        
        context = ' '.join(sentence[-self.n + 1:])
        for i in range(length):
            # word = max(self.prob(word, context) for word in self.ngrams[context])
            tmp = {word: self.prob(word, context) for word in self.ngrams[context]}
            word = max(tmp, key=tmp.get)

            if word == '</s>':
                break
            sentence.append(word)
            context = ' '.join(sentence[-self.n + 1:])
        if sentence[0] == '<s>':
            sentence = sentence[1:]
        return ' '.join(sentence)

def main(args):
    n = args.n
    sentences = [sent.text for sent in complete_doc.sents]
    ngrams = NGramModel(sentences, n, args.backoff, args.interpolation)
    print(ngrams.ngrams)
    print("Vocabs:", ngrams.v)
    print("Probabilities of word 'book' given context 'the' (no smoothing):", ngrams.prob("book", "the", laplace=0))
    print("Probabilities of word 'book' given context 'as' (no smoothing):", ngrams.prob("book", "as", laplace=0))
    print("Probabilities of word 'book' given context 'the' (smoothing factor = 1)", ngrams.prob("book", "the"))
    print("Probabilities of word 'book' given context 'as' (smoothing factor = 1)", ngrams.prob("book", "as"))
    print("Generation with context 'The book are':",ngrams.generate(context="The books are", length=10))

if __name__ == '__main__':
    # create argument parser object
    argumentParser = ArgumentParser(description='ngram model')
    argumentParser.add_argument('--n', type=int, default=2)
    argumentParser.add_argument('--backoff', action='store_true')
    argumentParser.add_argument('--interpolation', action='store_true')
    args = argumentParser.parse_args()
    main(args)

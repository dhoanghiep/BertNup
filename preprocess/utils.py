from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, f1_score, confusion_matrix, log_loss

class Sequence():
    def __init__(self, seq: str):
        self.seq = seq.upper()
    def __str__(self):
        return self.seq
    def __repr__(self):
        return f'<{type(self).__name__}: {self.seq}>'

class DNASequence(Sequence):
    def __init__(self, na_seq: str, original_bases = 'DNA'):
        super().__init__(na_seq)
        assert original_bases in ['DNA','RNA'], 'choose original bases: "DNA" or "RNA"'
        if original_bases == 'RNA':
            self.seq = self.seq.replace("U", "T")

    def to_kmer_sequence(self,k: int):
        assert k <= len(self), 'k is larger than sequence length'
        dna_seq = self.seq
        kmer_seq = []
        for i in range(len(self)):
            kmer_seq.append(dna_seq[i:i+k])
        kmer_seq = ' '.join(kmer_seq)
        return KmerSequence(kmer_seq)

    def reverse_complement(self):
        string = self.seq
        complement_string = str()
        for nu in string:
            if nu == 'A':
                complement_string += 'T'
            elif nu == 'T':
                complement_string += 'A'
            elif nu == 'G':
                complement_string += 'C'
            elif nu == 'C':
                complement_string += 'G'
            elif nu == 'N':
                complement_string += 'N'
        return DNASequence(complement_string[::-1])

    def __len__(self):
        return len(self.seq)

class KmerSequence(Sequence):
    def __init__(self, kmer_seq: str):
        super().__init__(kmer_seq)
    def to_dna_sequence(self):
        kmer_seq = self.seq
        dna_seq = ''.join([kmer[0] for kmer in kmer_seq.split(' ')])
        return DNASequence(dna_seq)

def compute_all_metrics(probas, labels, threshold = 0.5, verbose = 0):
    preds = (probas > threshold).astype(int)
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    auc = roc_auc_score(labels, probas)
    f1 = f1_score(labels, preds, zero_division = 0)
    tn, fp, fn, tp = confusion_matrix(labels, preds).flatten()
    sn = tp/(tp+fn)
    sp = tn/(tn+fp)
    if verbose == 1:
        loss = log_loss(labels, probas)
        print(f'loss: {loss:.6f}')
    return sn, sp, acc, f1, mcc, auc

"""DNA sequence representations and k-mer conversion."""

from __future__ import annotations


class Sequence:
    """Base sequence class."""

    def __init__(self, seq: str):
        self.seq = seq.upper()

    def __str__(self):
        return self.seq

    def __repr__(self):
        return f"<{type(self).__name__}: {self.seq}>"


class DNASequence(Sequence):
    """DNA sequence with k-mer conversion and reverse complement."""

    def __init__(self, na_seq: str, original_bases: str = "DNA"):
        super().__init__(na_seq)
        assert original_bases in ("DNA", "RNA"), 'choose original bases: "DNA" or "RNA"'
        if original_bases == "RNA":
            self.seq = self.seq.replace("U", "T")

    def to_kmer_sequence(self, k: int) -> KmerSequence:
        assert k <= len(self), "k is larger than sequence length"
        kmer_seq = " ".join(self.seq[i : i + k] for i in range(len(self)))
        return KmerSequence(kmer_seq)

    def reverse_complement(self) -> DNASequence:
        complement = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
        return DNASequence("".join(complement[nt] for nt in reversed(self.seq)))

    def __len__(self):
        return len(self.seq)


class KmerSequence(Sequence):
    """K-mer tokenized sequence."""

    def to_dna_sequence(self) -> DNASequence:
        dna_seq = "".join(kmer[0] for kmer in self.seq.split(" "))
        return DNASequence(dna_seq)

import torch
import esm
import re

esm_singleton = None

class ESMUtil:
    def __init__(self):
        global esm_singleton
        if esm_singleton is not None:
            return esm_singleton
        # Load ESM-2 model
        # TODO: Replace this model by esm2_t48_15B_UR50D() in production
        # TODO: Remember to change the number of layers as well
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = self.model.to(self.device)
        self.n_layers = 33
        self.batch_converter = self.alphabet.get_batch_converter()
        # disables dropout for deterministic results
        self.model.eval()
        self.batch_labels = self.batch_strs = self.batch_tokens = None
        esm_singleton = self
    
    def load_data(self, data):
        for i in range(len(data)):
            assert len(data[i]) == 2
            assert type(data[i][1]) == str
            data[i] = (data[i][0], re.sub(r'[^A-Z]', '', data[i][1]))
            assert not '[' in data[i][1]
            assert not ']' in data[i][1]
        self.batch_labels, self.batch_strs, self.batch_tokens = self.batch_converter(data)
        self.batch_tokens = self.batch_tokens.to(self.device)
        # For each sequence in a batch, collect its non-padding tokens
        self.batch_lens = (self.batch_tokens != self.alphabet.padding_idx).sum(1)
        return self

    def _get_token_representations(self):
        assert self.batch_labels is not None
        assert self.batch_strs is not None
        assert self.batch_tokens is not None
        # Extract per-residue representations
        with torch.no_grad():
            results = self.model(self.batch_tokens, repr_layers=[self.n_layers], return_contacts=False)
        token_reperesentations = results["representations"][self.n_layers].to(self.device)
        return token_reperesentations

    def get_sequence_representations(self):
        assert self.batch_labels is not None
        assert self.batch_strs is not None
        assert self.batch_tokens is not None
        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, tokens_len in enumerate(self.batch_lens):
            sequence_representations.append(self._get_token_representations()[i, 1 : tokens_len - 1].mean(0))
        return sequence_representations
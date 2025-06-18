class Tokenizer:
    def __init__(self, idx2token):
        self.idx2token = idx2token

    def decode(self, ids, eos_token="<eos>", pad_token="<pad>"):
        tokens = []
        for idx in ids:
            token = self.idx2token.get(idx, "<unk>")
            if token == eos_token:
                break
            if token == pad_token:
                continue
            tokens.append(token)
        return " ".join(tokens)

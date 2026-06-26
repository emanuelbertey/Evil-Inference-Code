"""BPE ByteLevel tokenizer — compatible with Rust Tokenizer::load_pretrained()."""


class BPEWrapper:
    def __init__(self, path_or_tokenizer):
        if isinstance(path_or_tokenizer, str):
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(path_or_tokenizer)
        else:
            self.tokenizer = path_or_tokenizer
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

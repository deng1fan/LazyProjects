import torch.nn as nn
import torch

class Pipeline(nn.Module):
    def __init__(self, model, tokenizer, config):
        super().__init__()
        self.model = model.to(config.default_device)
        self.tokenizer = tokenizer
        self.config = config

    def forward(self, input_ids=None, input_text=None, **other_features):
        if not input_ids:
            input_text = [input_text] if isinstance(input_text, str) else input_text
            input_ids = self.tokenizer.encode(input_text)
        input_ids = torch.LongTensor(input_ids)
        other_features['decoder_stage'] = 'test'
        generated_ids = self.model.generate(
                    input_ids=input_ids.to(self.config.default_device),
                    num_beams=self.config.beam_size,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    max_length=self.config.max_generation_length,
                    min_length=3,
                    **other_features
                )
        generated_sentences = [
            {'seqs': self.tokenizer.decode(sent, skip_special_tokens=True),
             'seqs_with_special_tokens': self.tokenizer.decode(sent, skip_special_tokens=False, ignore_tokens=[self.tokenizer.pad_token])
             } for i, sent in enumerate(generated_ids)]
        return generated_sentences
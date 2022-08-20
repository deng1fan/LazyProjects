import torch
from collections import defaultdict
from general_files.utils.others.bertscore.utils import (
    get_model,
    get_tokenizer,
    get_idf_dict,
    bert_cos_score_idf,
    lang2model,
    model2layers,
)
import sys
import os
import pandas as pd

class BertScorer:
    def init_scorer(
            self,
            model_type=None,
            num_layers=None,
            device=None,
            nthreads=4,
            all_layers=False,
            lang=None,
            batch_size=64,
            use_fast_tokenizer=False,
            idf=False,
            baseline_path=None,
            rescale_with_baseline=True,
    ):
        """
        BERTScore metric.

        Args:
            - :param: `cands` (list of str): candidate sentences
            - :param: `refs` (list of str or list of list of str): reference sentences
            - :param: `model_type` (str): bert specification, default using the suggested
                      model for the target langauge; has to specify at least one of
                      `model_type` or `lang`
            - :param: `num_layers` (int): the layer of representation to use.
                      default using the number of layer tuned on WMT16 correlation data
            - :param: `verbose` (bool): turn on intermediate status update
            - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
            - :param: `device` (str): on which the contextual embedding model will be allocated on.
                      If this argument is None, the model lives on cuda:0 if cuda is available.
            - :param: `nthreads` (int): number of threads
            - :param: `batch_size` (int): bert score processing batch size
            - :param: `lang` (str): language of the sentences; has to specify
                      at least one of `model_type` or `lang`. `lang` needs to be
                      specified when `rescale_with_baseline` is True.
            - :param: `return_hash` (bool): return hash code of the setting
            - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
            - :param: `baseline_path` (str): customized baseline file
            - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer

        Return:
            - :param: `(P, R, F)`: each is of shape (N); N = number of input
                      candidate reference pairs. if returning hashcode, the
                      output will be ((P, R, F), hashcode). If a candidate have
                      multiple references, the returned score of this candidate is
                      the *best* score among all references.
        """
        self.all_layers = all_layers
        self.batch_size = batch_size
        self.nthreads = nthreads
        self.idf = idf
        self.lang = lang
        self.model_type = model_type
        self.num_layers = num_layers
        self.rescale_with_baseline = rescale_with_baseline
        self.baseline_path = baseline_path
        if model_type is None:
            lang = lang.lower()
            self.model_type = lang2model[lang]
        if num_layers is None:
            self.num_layers = model2layers[self.model_type]

        self.tokenizer = get_tokenizer(self.model_type, use_fast_tokenizer)
        self.model = get_model(self.model_type, self.num_layers, self.all_layers)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def get_score(self, refs, cands):
        assert len(cands) == len(refs), "Different number of candidates and references"
        ref_group_boundaries = None
        if not isinstance(refs[0], str):
            ref_group_boundaries = []
            ori_cands, ori_refs = cands, refs
            cands, refs = [], []
            count = 0
            for cand, ref_group in zip(ori_cands, ori_refs):
                cands += [cand] * len(ref_group)
                refs += ref_group
                ref_group_boundaries.append((count, count + len(ref_group)))
                count += len(ref_group)
        if not self.idf:
            idf_dict = defaultdict(lambda: 1.0)
            # set idf for [SEP] and [CLS] to 0
            idf_dict[self.tokenizer.sep_token_id] = 0
            idf_dict[self.tokenizer.cls_token_id] = 0
        elif isinstance(self.idf, dict):
            idf_dict = self.idf
        else:
            idf_dict = get_idf_dict(refs, self.tokenizer, nthreads=self.nthreads)

        all_preds = bert_cos_score_idf(
            self.model,
            refs,
            cands,
            self.tokenizer,
            idf_dict,
            verbose=False,
            device='cuda:0',
            batch_size=self.batch_size,
            all_layers=self.all_layers,
        ).cpu()

        if ref_group_boundaries is not None:
            max_preds = []
            for beg, end in ref_group_boundaries:
                max_preds.append(all_preds[beg:end].max(dim=0)[0])
            all_preds = torch.stack(max_preds, dim=0)

        if self.rescale_with_baseline:
            if self.baseline_path is None:
                baseline_path = os.path.join(os.path.dirname(__file__), f"rescale_baseline/{self.lang}/{self.model_type}.tsv")
            if os.path.isfile(baseline_path):
                if not self.all_layers:
                    baselines = torch.from_numpy(pd.read_csv(baseline_path).iloc[self.num_layers].to_numpy())[1:].float()
                else:
                    baselines = torch.from_numpy(pd.read_csv(baseline_path).to_numpy())[:, 1:].unsqueeze(1).float()

                all_preds = (all_preds - baselines) / (1 - baselines)
            else:
                print(
                    f"Warning: Baseline not Found for {self.model_type} on {self.lang} at {baseline_path}", file=sys.stderr,
                )

        out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F
        return out
import sys
from pathlib import Path
from typing import override
import numpy as np
import torch
from torch import nn


sys.path.insert(0, "/projectnb/vtrs/joseff/mpra-predictor")
import mpra_predictor


VALID_BASES = {'A', 'C', 'G', 'T'}
BATCH_SIZE = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PARAM_CONFIG = {
    'sei_pooler': {'n_heads': 1, 'hidden_dim': 128, 'pos_emb_dim': 8, 'dropout': 0.5, },
    'mal_pooler': {'n_heads': 1, 'hidden_dim': 64, 'pos_emb_dim': 8, 'dropout': 0.5, },
    'fusion': {'output_dim': 128, 'dropout': 0.7},
    'mlp': {'hidden_size': 128, 'num_res_blocks': 0, 'dropout': 0.5},
    'target_columns': [
        'GM12878_mean', 'Jurkat_mean', 'MRC5_mean', 'A549_mean',
        'HEK293T_mean', 'K562_mean', 'SHSY5Y_mean', 'SiHa_mean'
    ],
}
CREST_LABELS = np.array([col.removesuffix("_mean") for col in PARAM_CONFIG["target_columns"]])
CREST_DIR = Path("/projectnb/vtrs/joseff/mpra-predictor")


class PredictorBase():
    def __init__(self, model: nn.Module, BATCH_SIZE: int, needs_validation: bool = True):
        self.model = model.to(DEVICE).eval()
        self.BATCH_SIZE = BATCH_SIZE
        self.valid_ids = []
        self.batch_sequences = []
        self.predictions = []
        self.needs_validation = needs_validation

    @staticmethod
    def validate_tile_sequence(tile_sequence: str) -> bool:
        if len(tile_sequence) != 200:
            return False
        if len(set(tile_sequence) - VALID_BASES) != 0:
            return False
        return True

    def update(self, tile_id: str, tile_sequence: str) -> None:
        if self.needs_validation and not self.validate_tile_sequence(tile_sequence):
            return
        self.valid_ids.append(tile_id)
        self.batch_sequences.append(tile_sequence)
        if len(self.batch_sequences) == self.BATCH_SIZE:
            self.predictions.append(self.batch_predict(self.batch_sequences))
            self.batch_sequences.clear()

    def get_predictions(self,) -> np.ndarray:
        if self.batch_sequences:
            self.predictions.append(self.batch_predict(self.batch_sequences))
            self.batch_sequences.clear()
        valid_ids = np.array(self.valid_ids)
        predictions = np.concatenate(self.predictions)
        return valid_ids, predictions

    def batch_predict(self, batch_sequences: list[str]) -> np.ndarray:
        pass


class CREST(PredictorBase):
    def __init__(self, batch_size: int):
        self.sei_flank_builder = mpra_predictor.dataloader.prepare_flank_builder(input_size=4096)
        self.mal_flank_builder = mpra_predictor.dataloader.prepare_flank_builder(input_size=600)
        model = mpra_predictor.load.load_model_structure(PARAM_CONFIG, load_pretrained_base_weights=True).to(DEVICE)
        model.load_state_dict(torch.load(CREST_DIR / f"data/trained_models/{str(model)}.pt", map_location=DEVICE), strict=True)
        super().__init__(model, batch_size)

    @override
    def batch_predict(self, batch_sequences: list[str]) -> np.ndarray:
        onehots = [mpra_predictor.dataloader.dna_to_tensor(seq) for seq in batch_sequences]
        onehots_sei = self.sei_flank_builder.add_flanks(onehots).to(DEVICE)
        onehots_mal = self.mal_flank_builder.add_flanks(onehots).to(DEVICE)
        with torch.no_grad():
            preds = self.model(onehots_sei, onehots_mal)
        return preds.cpu().numpy()


class CRESTInterpreter:
    def __init__(self, batch_size: int, pred_index: int = 4):
        # NOTE: by default pred_idx = 4, which corresponds to HEK293T MPRA prediction
        self.batch_size = batch_size
        self.batch_sequences = list()
        self.valid_ids = list()
        self.onehots = list()
        self.contribs = list()

        self.sei_flank_builder = mpra_predictor.dataloader.prepare_flank_builder(input_size=4096)
        self.mal_flank_builder = mpra_predictor.dataloader.prepare_flank_builder(input_size=600)
        model = mpra_predictor.load.load_model_structure(PARAM_CONFIG, load_pretrained_base_weights=True).to(DEVICE)
        model.load_state_dict(torch.load(CREST_DIR / f"data/trained_models/{str(model)}.pt", map_location=DEVICE), strict=True)
        self.predictor = mpra_predictor.interpret.Predictor(model, pred_idx=pred_index).to(DEVICE).eval()

    def update(self, tile_id: str, tile_sequence: str) -> None:
        if not self.validate_tile_sequence(tile_sequence):
            return
        self.valid_ids.append(tile_id)
        self.batch_sequences.append(tile_sequence)
        if len(self.batch_sequences) == self.batch_size:
            self.contribs.append(self.batch_infer(self.batch_sequences))
            self.batch_sequences.clear()

    @staticmethod
    def validate_tile_sequence(tile_sequence: str) -> bool:
        if len(tile_sequence) != 200:
            return False
        if len(set(tile_sequence) - VALID_BASES) != 0:  # invalid sequence
            return False
        return True

    def batch_infer(self, batch_sequences: list[str]) -> np.ndarray:
        onehots = [mpra_predictor.dataloader.dna_to_tensor(seq) for seq in batch_sequences]
        self.onehots.append(onehots)
        onehots_sei = self.sei_flank_builder.add_flanks(onehots).to(DEVICE)
        onehots_mal = self.mal_flank_builder.add_flanks(onehots).to(DEVICE)
        contribs_lst = mpra_predictor.interpret.isg_contributions_multi_input(
            [onehots_sei, onehots_mal], self.predictor, num_steps=50, step_chunk_size=10, use_tqdm=False
        )
        contribs_sei = contribs_lst[0][..., 1948: 2148]
        contribs_mal = contribs_lst[1][..., 200: 400]
        contribs = contribs_sei + contribs_mal
        return contribs.cpu().numpy().astype(np.float32)

    def get_predictions(self,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.batch_sequences:
            self.contribs.append(self.batch_infer(self.batch_sequences))
            self.batch_sequences.clear()
        valid_ids = np.array(self.valid_ids)
        onehots = np.concatenate(self.onehots)
        contribs = np.concatenate(self.contribs)
        return valid_ids, onehots, contribs

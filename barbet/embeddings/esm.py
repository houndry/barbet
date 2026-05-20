from enum import Enum
import typer
from pathlib import Path
import torch
from torchapp.cli import method
from barbet.embedding import Embedding



class ESMLayers(Enum):
    T6 = "6"
    T12 = "12"
    T30 = "30"
    T33 = "33"
    T36 = "36"
    T48 = "48"

    @classmethod
    def from_value(cls, value: int|str) -> "ESMLayers":
        for layer in cls:
            if layer.value == str(value):
                return layer
        return None
    
    def __int__(self):
        return int(self.value)
    
    def __str__(self):
        return str(self.value)

    def model_name(self) -> str:
        match self:
            case ESMLayers.T48:
                return "esm2_t48_15B_UR50D"
            case ESMLayers.T36:
                return "esm2_t36_3B_UR50D"
            case ESMLayers.T33:
                return "esm2_t33_650M_UR50D"
            case ESMLayers.T30:
                return "esm2_t30_150M_UR50D"
            case ESMLayers.T12:
                return "esm2_t12_35M_UR50D"
            case ESMLayers.T6:
                return "esm2_t6_8M_UR50D"

    def get_model_alphabet(self) -> tuple["ESM2", "Alphabet"]:
        return torch.hub.load("facebookresearch/esm:main", self.model_name(), verbose=False)


class ESMEmbedding(Embedding):
    @method
    def setup(
        self, 
        layers:ESMLayers=typer.Option(ESMLayers.T6, help="The number of ESM layers to use."),
        hub_dir:Path=typer.Option(None, help="The torch hub directory where the ESM models will be cached."),
    ):
        if layers and not getattr(self, 'layers', None):
            self.layers = layers

        if isinstance(self.layers, (str,int)):
            self.layers = ESMLayers.from_value(self.layers)
        
        assert self.layers is not None, f"Please ensure the number of ESM layers is one of " + ", ".join(ESMLayers.keys())
        assert isinstance(self.layers, ESMLayers)

        self.hub_dir = hub_dir
        if hub_dir:
            torch.hub.set_dir(str(hub_dir))
        self.model = None
        self.device = None
        self.batch_converter = None
        self.alphabet = None

    def __getstate__(self):
        state = super().__getstate__()
        state.update(dict(layers=str(self.layers)))
        return state

    def __setstate__(self, state):
        self.__init__()

        # Restore the object state from the unpickled state
        self.__dict__.update(state)
        self.model = None
        self.device = None
        self.batch_converter = None
        self.alphabet = None
        self.hub_dir = None

    def load(self):
        self.model, self.alphabet = self.layers.get_model_alphabet()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def embed(self, seq:str) -> torch.Tensor:
        """ Takes a protein sequence as a string and returns an embedding tensor per residue. """
        if isinstance(self.layers, (str,int)):
            self.layers = ESMLayers.from_value(self.layers)
        
        layers = int(self.layers.value)

        # Handle ambiguous AAs
        # https://github.com/facebookresearch/esm/issues/164
        seq = seq.replace("J", "X")

        if not self.model:
            self.load()

        _, _, batch_tokens = self.batch_converter([("marker_id", seq)])
        batch_tokens = batch_tokens.to(self.device)                
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[layers], return_contacts=True)
        token_representations = results["representations"][layers]

        assert len(batch_lens) == 1, f"More than one length found"
        assert token_representations.size(0) == 1, f"More than one representation found"

        # Strip off the beginning-of-sequence and end-of-sequence tokens
        embedding_tensor = token_representations[0, 1 : batch_lens[0] - 1]
        assert len(seq) == len(embedding_tensor), f"Embedding representation incorrect length. should be {len(seq)} but is {len(embedding_tensor)}"

        return embedding_tensor
    
    # def embed_batch(self, seqs:list[str]) -> torch.Tensor:
    #     """ Takes a list of protein sequences and returns a tensor of embeddings per residue. """
    #     if isinstance(self.layers, (str,int)):
    #         self.layers = ESMLayers.from_value(self.layers)
        
    #     layers = int(self.layers.value)

    #     # Handle ambiguous AAs
    #     #

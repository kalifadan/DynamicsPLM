import esm
import itertools
import torch

from esm.model.esm2 import ESM2
from utils.constants import foldseek_seq_vocab, foldseek_struc_vocab


def load_protligand(model_path: str, ligand_generator_path: str):
    """
    Load ProtLigand model.
    Args:
        model_path: path to ProtLigand model
        ligand_generator_path: path to the Ligand Generator Model
    """
    
    # Initialize the alphabet
    tokens = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
    for seq_token, struc_token in itertools.product(foldseek_seq_vocab, foldseek_struc_vocab):
        token = seq_token + struc_token
        tokens.append(token)
    
    alphabet = esm.data.Alphabet(standard_toks=tokens,
                                 prepend_toks=[],
                                 append_toks=[],
                                 prepend_bos=True,
                                 append_eos=True,
                                 use_msa=False)
    
    alphabet.all_toks = alphabet.all_toks[:-2]
    alphabet.unique_no_split_tokens = alphabet.all_toks
    alphabet.tok_to_idx = {tok: i for i, tok in enumerate(alphabet.all_toks)}
    
    # Load weights
    data = torch.load(model_path)
    weights = data["model"]
    config = data["config"]
    
    # Initialize the model
    model = ESM2(
        num_layers=config["num_layers"],
        embed_dim=config["embed_dim"],
        attention_heads=config["attention_heads"],
        alphabet=alphabet,
        token_dropout=config["token_dropout"],
    )
    
    load_weights(model, weights)




    return model, alphabet


def load_weights(model, weights):
    model_dict = model.state_dict()

    # Load additional modules dynamically
    for key, module in [
        ("cross_attention", model.cross_attention),
        ("ligand_proj", model.ligand_proj),
        ("ligand_generator", model.ligand_generator),
        ("ligand_decoder", model.ligand_decoder)
    ]:
        if key in state_dict:
            module.load_state_dict(state_dict[key])

    unused_params = []
    missed_params = list(model_dict.keys())

    for k, v in weights.items():
        if k in model_dict.keys():
            model_dict[k] = v
            missed_params.remove(k)

        else:
            unused_params.append(k)

    if len(missed_params) > 0:
        print(f"\033[31mSome weights of {type(model).__name__} were not "
              f"initialized from the model checkpoint: {missed_params}\033[0m")

    if len(unused_params) > 0:
        print(f"\033[31mSome weights of the model checkpoint were not used: {unused_params}\033[0m")

    model.load_state_dict(model_dict)
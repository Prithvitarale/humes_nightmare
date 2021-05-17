import argparse
import code
from importlib import reload
import json
import logging
import numpy as np
import re
import typing
from typing import List, Tuple, Set, Dict, Any, Optional, NamedTuple, Iterator
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.WARN,
)

logger = logging.getLogger(__name__)

import pandas as pd
import torch
import tqdm

from sc.model.common import init_model, load_data, get_all_attributes
from sc.examples import examples

MFS = (
    "Moral Foundations",
    [
        "care-harm",
        "fairness-cheating",
        "loyalty-betrayal",
        "authority-subversion",
        "sanctity-degradation",
    ],
)

CATS = (
    "Rule-of-Thumb Category",
    [
        "morality-ethics",
        "social-norms",
        "advice",  # rarely want these
        # "description",  # off by default, as very rarely want these
    ],
)

AGREEMENTS = (
    "How many people probably agree with this ROT?",
    [
        # People basically didn't write ROTs for these categories. (Which is expected.)
        # "nobody",
        # "rare",
        # "controversial",
        # "most",
        "all",
    ],
)

PRESSURES = (
    "How much cultural pressure do you feel to do the action?",
    ["strong-against", "against", "discretionary", "for", "strong-for"],
)

MORAL_JUDGMENTS = (
    "How good/bad is it to do the action morally?",
    ["very-bad", "bad", "ok", "good", "very-good"],
)

LEGAL = ("Is the action legal?", ["illegal", "tolerated", "legal"])


Examples = List[Tuple[str, Tuple[str, List[str]]]]


def main(ex):
    """
    Generate ea RoT for each situation the text
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument(
        "--model",
        type=str,
        help="LM checkpoint for initialization.",
    )

    # Optional
    parser.add_argument(
        "--max_length", default=40, type=int, required=False, help="Maximum text length"
    )
    parser.add_argument(
        "--k", default=0, type=int, required=False, help="k for top k sampling"
    )
    parser.add_argument(
        "--do_lower_case", default=True, type=bool, required=False, help="lower?"
    )
    # Using p=0.9 by default
    parser.add_argument(
        "--p", default=0.9, type=float, required=False, help="p for nucleus sampling"
    )
    parser.add_argument(
        "--beams", default=0, type=int, required=False, help="beams for beam search"
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        required=False,
        help="temperature for sampling",
    )
    parser.add_argument("--device", default="0", type=str, help="GPU number or 'cpu'.")
    args = parser.parse_args()

    if (
        (args.k == args.p == args.beams == 0)
        or (args.k != 0 and args.p != 0)
        or (args.beams != 0 and args.p != 0)
        or (args.beams != 0 and args.k != 0)
    ):
        raise ValueError(
            "Exactly one of p, k, and beams should be set to a non-zero value."
        )

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    args.device = device

    args.model = "..\\..\\gpt2-xl_rot_64_5epochs"
    mode = 'action' if 'action' in args.model else 'rot'
    # mode = 'rot'

    tokenizer, model = init_model(args.model, args)
    eos_token_id = tokenizer.encode("<eos>", add_special_tokens=False)[0]

    # just caching these for now
    special_tokens = [
        "<controversial>",
        "<against>",
        "<for>",
        "<description>",
        "<legal>",
        "<care-harm>",
        "<explicit-no>",
        "<loyalty-betrayal>",
        "<explicit>",
        "<all>",
        "<strong-against>",
        "<morality-ethics>",
        "<good>",
        "<probable>",
        "<social-norms>",
        "<authority-subversion>",
        "<most>",
        "<ok>",
        "<hypothetical>",
        "<discretionary>",
        "<fairness-cheating>",
        "<bad>",
        "<agency>",
        "[attrs]",
        "[rot]",
        "[action]",
        "<eos>",
    ]
    preds = []
    while True:
        for prompt, (variable, options) in ex:
            n_gens = 5
            for option in options:
                input_ = prompt.format(varies=option)
                input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_))
                input_length = len(input_ids)
                max_length = args.max_length + input_length
                input_ids = torch.tensor([input_ids]).to(device)

                # NOTE: swap this out for newer model
                eos_token_id = tokenizer.encode("<eos>", add_special_tokens=False)[0]
                # eos_token_id = tokenizer.eos_token_id,
                pred_category = []
                for _ in range(n_gens):

                    outputs = model.generate(
                        input_ids,
                        do_sample=args.beams == 0,
                        max_length=max_length,
                        temperature=args.temperature,
                        top_p=args.p if args.p > 0 else None,
                        top_k=args.k if args.k > 0 else None,
                        eos_token_id=eos_token_id,
                        num_beams=args.beams if args.beams > 0 else None,
                        early_stopping=True,
                        # NOTE: swap for new models that (I think) actually have pad id set.
                        # pad_token_id=tokenizer.pad_token_id,
                        pad_token_id=50256,
                        no_repeat_ngram_size=3,
                    )

                    pred = tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True,
                        # clean_up_tokenization_spaces=False,
                    )[len(input_):].strip()
                    pred_category.append(pred)
                preds.append(pred_category)
        break
    print(preds)
    return preds


def situation_to_attr(situation, axis, all_rot_vecs, model):
    if axis=="MFS":
        axis = MFS
    elif axis=="LEGAL":
        axis=LEGAL
    elif axis=="MORAL_JUDGMENTS":
        axis=MORAL_JUDGMENTS
    elif axis=="PRESSURES":
        axis=PRESSURES
    elif axis=="CATS":
        axis=CATS
    situation_formatted = (situation + " [attrs] <social-norms> <{varies}> [action]", axis),
    axis_rots = main(situation_formatted)
    best_category = 0
    most_similar = 0
    for i, category in enumerate(axis_rots):
        for rot in category:
            rot_vec = model.encode([rot])[0]
            rot_sim = max(cosine_similarity(rot_vec, all_rot_vec) for all_rot_vec in all_rot_vecs)
            if rot_sim >= most_similar:
                most_similar = rot_sim
                best_category = i
    # print(best_category)
    return best_category

def situation_to_attrs(situation):
    model = SentenceTransformer("bert-base-nli-mean-tokens")
    situation_formatted = (situation + " [attrs] <social-norms> <{varies}> [action]", AGREEMENTS),
    all_rots = main(situation_formatted)
    all_rot_vecs = [model.encode([rot])[0] for rot in all_rots[0]]
    attrs = np.zeros(5)
    attrs[0] = situation_to_attr(situation, "MFS", all_rot_vecs, model)
    attrs[1] = situation_to_attr(situation, "LEGAL", all_rot_vecs, model)
    attrs[2] = situation_to_attr(situation, "MORAL_JUDGMENTS", all_rot_vecs, model)
    attrs[3] = situation_to_attr(situation, "PRESSURES", all_rot_vecs, model)
    attrs[4] = situation_to_attr(situation, "CATS", all_rot_vecs, model)
    print(attrs)
    return attrs


def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


situation_to_attrs("cheating on your partner")

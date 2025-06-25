from __future__ import annotations

import multiprocessing as mp
import os
from collections import Counter

import regex as re

from .pretokenization_example import find_chunk_boundaries

END_OF_TEXT = "<|endoftext|>"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_RESERVED_TOKENS = 256
NUM_PROCESSES = 10

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {i: i.to_bytes() for i in range(NUM_RESERVED_TOKENS)}
    num_vocab = NUM_RESERVED_TOKENS
    for special_token in special_tokens:
        vocab[num_vocab] = special_token.encode("utf-8")
        num_vocab += 1
    assert len(vocab) == num_vocab

    # pre-tokenize

    bytes_pretoken_counter = get_pretoken_counter(input_path)
    # use the regex to split the input text

    # print(bytes_pretoken_counter)

    # pair counting: we need some termination condition for this
    while num_vocab < vocab_size:
        # get the most frequent (and lexicographically largest) pair
        byte_pair_cts = count_byte_pairs(bytes_pretoken_counter)
        top_pair: tuple[bytes, bytes] = get_top_pair(byte_pair_cts)
        # print(f"top pair: {top_pair}")

        # in bytes_pretoken_counter, for pretokens that have this pair, merge those pairs
        # (will have to break open a tuple then reconstruct a tuple)
        current_pretokens: list[tuple[bytes, ...]] = list(bytes_pretoken_counter.keys())
        for bytes_pretoken in current_pretokens:
            ct = bytes_pretoken_counter[bytes_pretoken]
            new_bytes_pretoken = merge(bytes_pretoken, top_pair)
            del bytes_pretoken_counter[bytes_pretoken]
            bytes_pretoken_counter[new_bytes_pretoken] = ct

        merges.append(top_pair)
        vocab[num_vocab] = b"".join(top_pair)
        num_vocab += 1

    return (vocab, merges)

def get_bytes_tuple(input_str: str) -> tuple[bytes, ...]:
    """Convert a string into a tuple of individual byte objects.

    Takes a UTF-8 string and converts it to a sequence of bytes, where each
    byte is represented as a separate bytes object in a tuple. This representation
    is used in BPE training to work with individual bytes as atomic units.

    Args:
        input_str: The input string to convert to bytes

    Returns:
        tuple[bytes, ...]: A tuple where each element is a single-byte bytes object
        representing one UTF-8 encoded byte from the input string

    Example:
        >>> get_bytes_tuple("hi")
        (b'h', b'i')
        >>> get_bytes_tuple("🙂")  # Multi-byte UTF-8 character
        (b'\xf0', b'\x9f', b'\x99', b'\x82')
    """
    utf8_encoded: bytes = input_str.encode("utf-8")
    # when you unpack a bytes object you get list of ints. we have to convert back manually to bytes
    return tuple(map(lambda i: i.to_bytes(), utf8_encoded))

def count_byte_pairs(bytes_pretoken_counter: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    """Count all consecutive byte pairs in pretokens.

    Analyzes each pretoken in the counter and counts how frequently each consecutive
    pair of bytes occurs across all pretokens, weighted by their frequencies.

    Args:
        bytes_pretoken_counter: Counter mapping byte sequences (as tuples) to their frequencies

    Returns:
        Counter[tuple[bytes, bytes]]: Counter mapping consecutive byte pairs to their
        total occurrence frequencies across all pretokens
    """
    byte_pair_cts: Counter[tuple[bytes, bytes]] = Counter()

    # count pairs of bytes
    for bytes_pretoken, ct in bytes_pretoken_counter.items():
        # for each bytes pretoken traverse the sequence, and update pair_cts accordingly
        for left_idx in range(len(bytes_pretoken) - 1):
            byte_pair: tuple[bytes, bytes] = (bytes_pretoken[left_idx], bytes_pretoken[left_idx + 1])
            byte_pair_cts[byte_pair] += ct

    return byte_pair_cts

def get_top_pair(byte_pair_cts: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
    """Find the most frequently occurring pair of consecutive bytes in the pretokens.

    Identifies the byte pair with the highest frequency count. In case of ties,
    returns the lexicographically largest pair (using byte-wise comparison).

    Args:
        byte_pair_cts: Counter mapping consecutive byte pairs to their frequencies

    Returns:
        tuple[bytes, bytes]: The most frequent pair of consecutive bytes. In case of ties,
        returns the lexicographically largest pair.
    """

    # get the most freq occuring one. if tie, lexicographically larger
    most_freq_bp: tuple[bytes, bytes] = (b"a", b"b")
    largest_bp_count = 0

    for byte_pair, ct in byte_pair_cts.items():
        # new king if larger count
        if ct > largest_bp_count:
            most_freq_bp, largest_bp_count = byte_pair, ct
        # if same count, only update if lexicographically larger
        elif ct == largest_bp_count and byte_pair > most_freq_bp:
            most_freq_bp, largest_bp_count = byte_pair, ct

    return most_freq_bp

def merge(bytes_pretoken: tuple[bytes, ...], new_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Merge all consecutive occurrences of a specific byte pair in a byte sequence.

    Args:
        bytes_pretoken: A tuple of bytes representing a pretoken sequence
        new_pair: A tuple of two bytes to be merged when found consecutively

    Returns:
        tuple[bytes, ...]: A new tuple with all consecutive occurrences of new_pair
        merged into single bytes objects (concatenated together)
    """
    new_bytes_list = []
    i = 0
    while i < len(bytes_pretoken):
        if i + 1 < len(bytes_pretoken) and (bytes_pretoken[i], bytes_pretoken[i + 1]) == new_pair:
            new_bytes_list.append(b"".join(new_pair))
            i += 2
        else:
            new_bytes_list.append(bytes_pretoken[i])
            i += 1
    return tuple(new_bytes_list)

def get_pretoken_counter(input_path: str | os.PathLike) -> Counter[tuple[bytes, ...]]:
    """Extract pretoken counts from input file using parallel processing.

    Splits the input file into chunks, processes each chunk in parallel to find
    pretokens using regex pattern matching, and combines the results into a single
    counter of byte-level pretokens.

    Args:
        input_path: Path to the input text file to process

    Returns:
        Counter[tuple[bytes, ...]]: A counter mapping byte pretoken tuples to their
        frequencies across the entire input file
    """

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, END_OF_TEXT.encode("utf-8"))
        args = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with mp.Pool(NUM_PROCESSES) as pool:
        all_pretoken_counters = pool.starmap(get_pretoken_counter_one_chunk, args)

    final_counter: Counter[tuple[bytes, ...]] = Counter()
    for pretoken_counter in all_pretoken_counters:
        final_counter.update(pretoken_counter)
    return final_counter

def get_pretoken_counter_one_chunk(input_path: str | os.PathLike, start: int, end: int) -> Counter[tuple[bytes, ...]]:
    """Process a single chunk of the input file to extract pretoken counts.

    Reads a specific byte range from the input file, applies regex pattern matching
    to find pretokens, and converts them to byte-level representations.

    Args:
        input_path: Path to the input text file to process
        start: Starting byte position in the file
        end: Ending byte position in the file (exclusive)

    Returns:
        Counter[tuple[bytes, ...]]: Counter of byte pretoken tuples found in this chunk
    """
    str_pretoken_counter: Counter[str] = Counter()
    bytes_pretoken_counter: Counter[tuple[bytes, ...]] = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        raw_chunk = f.read(end - start).decode("utf-8", errors="ignore")
        raw_chunk_split_by_eot = re.split(re.escape(END_OF_TEXT), raw_chunk)
        chunk = "|".join(raw_chunk_split_by_eot)

        # import pytest; pytest.set_trace()
        for pretoken in re.finditer(PAT, chunk):
            # import pytest; pytest.set_trace()
            pretoken_str = pretoken.group(0)
            if pretoken_str:
                str_pretoken_counter[pretoken_str] += 1

        bytes_pretoken_counter = Counter({get_bytes_tuple(pretoken_str): ct for pretoken_str, ct in str_pretoken_counter.items()})
    return bytes_pretoken_counter
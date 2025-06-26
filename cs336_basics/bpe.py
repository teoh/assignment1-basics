from __future__ import annotations

import multiprocessing as mp
import os
from collections import Counter, defaultdict

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

    # convert to a format that's easier to work with for caching counts
    idx_to_count: Counter[int] = Counter({idx: count for idx, count in enumerate(bytes_pretoken_counter.values())})
    idx_to_pretoken: dict[int, tuple[bytes, ...]] = {idx: pretoken for idx, pretoken in enumerate(bytes_pretoken_counter.keys())}
    bytes_pair_to_count, bytes_pair_to_idxs = count_byte_pairs(idx_to_count, idx_to_pretoken)
    # use the regex to split the input text

    # print(bytes_pretoken_counter)
    # import pytest; pytest.set_trace()

    # pair counting: we need some termination condition for this
    while num_vocab < vocab_size:
        # get the most frequent (and lexicographically largest) pair
        top_pair: tuple[bytes, bytes] = get_top_pair(bytes_pair_to_count)
        top_pair_bytes: bytes = b"".join(top_pair)

        # remove this pair from the pair counts. it is now its own unit
        pretoken_idxs_with_top_pair = bytes_pair_to_idxs[top_pair]
        del bytes_pair_to_count[top_pair]
        del bytes_pair_to_idxs[top_pair]

        for pretoken_idx in pretoken_idxs_with_top_pair:
            # for each pretoken that contained the top pair:
            # merge, and incrementally update counts of pairs
            pretoken_bytes: tuple[bytes, ...] = idx_to_pretoken[pretoken_idx]
            pretoken_ct = idx_to_count[pretoken_idx]
            new_bytes_list = []
            i = 0
            while i < len(pretoken_bytes):
                # found the pair in the seq: do the merge
                if i + 1 < len(pretoken_bytes) and (pretoken_bytes[i], pretoken_bytes[i+1]) == top_pair:
                    # imagine your top pair is (X, Y), sequence looks like: A, X, Y, B
                    # then merge results is A, XY, B
                    # we need to decrement counts of pairs: (A, X), (Y, B)
                    # and increment counts of: (A, XY), (XY, B)
                    new_bytes_list.append(top_pair_bytes)

                    # check left side of the pair
                    if i > 0:
                        # decrement (A, X)
                        bytes_pair_to_count[(pretoken_bytes[i-1], pretoken_bytes[i])] -= pretoken_ct
                        # increment (A, XY)
                        bytes_pair_to_count[(pretoken_bytes[i-1], top_pair_bytes)] += pretoken_ct
                        bytes_pair_to_idxs[(pretoken_bytes[i-1], top_pair_bytes)].add(pretoken_idx)

                    # check right side of the pair
                    if i + 2 < len(pretoken_bytes):
                        # decrement (Y, B)
                        bytes_pair_to_count[(pretoken_bytes[i+1], pretoken_bytes[i+2])] -= pretoken_ct
                        # increment (XY, B)
                        bytes_pair_to_count[(top_pair_bytes, pretoken_bytes[i+2])] += pretoken_ct
                        bytes_pair_to_idxs[(top_pair_bytes, pretoken_bytes[i+2])].add(pretoken_idx)

                    i += 2
                else:
                    new_bytes_list.append(pretoken_bytes[i])
                    i += 1
            idx_to_pretoken[pretoken_idx] = tuple(new_bytes_list)

        # do the merge step: for pretokens (i.e. bytes tuples) that contain this
        # new pair, perform the merge operation
        # e.g. you merged bytes (X, Y) -> now is XY
        # while in merge step, you encounter A, X, Y, B. your merge is now A, XY, B.
        # decrement pairs (A, X) and (Y, B), increment pairs (A, XY), (XY, B)

        # old
        # # print(f"top pair: {top_pair}")

        # # in bytes_pretoken_counter, for pretokens that have this pair, merge those pairs
        # # (will have to break open a tuple then reconstruct a tuple)
        # current_pretokens: list[tuple[bytes, ...]] = list(bytes_pretoken_counter.keys())
        # for bytes_pretoken in current_pretokens:
        #     ct = bytes_pretoken_counter[bytes_pretoken]
        #     new_bytes_pretoken = merge(bytes_pretoken, top_pair)
        #     del bytes_pretoken_counter[bytes_pretoken]
        #     bytes_pretoken_counter[new_bytes_pretoken] = ct

        merges.append(top_pair)
        vocab[num_vocab] = top_pair_bytes
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

def count_byte_pairs(idx_to_count: Counter[int], idx_to_pretoken: dict[int, tuple[bytes, ...]]) -> tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[int]]]:
    """Count all consecutive byte pairs in pretokens and track which pretokens contain each pair.

    Analyzes each pretoken and counts how frequently each consecutive pair of bytes
    occurs across all pretokens, weighted by their frequencies. Also tracks which
    pretoken indices contain each byte pair for efficient updates during merging.

    Args:
        idx_to_count: Counter mapping pretoken indices to their occurrence counts
        idx_to_pretoken: Dictionary mapping pretoken indices to their byte sequences

    Returns:
        tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[int]]]:
            bytes_pair_to_count: Counter mapping consecutive byte pairs to their total frequencies
            bytes_pair_to_idxs: Dictionary mapping byte pairs to sets of pretoken indices that contain them
    """
    bytes_pair_to_count: Counter[tuple[bytes, bytes]] = Counter()
    bytes_pair_to_idxs: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    # count pairs of bytes
    for idx, bytes_pretoken in idx_to_pretoken.items():
        ct = idx_to_count[idx]
        # for each bytes pretoken traverse the sequence, and update pair_cts accordingly
        for left_idx in range(len(bytes_pretoken) - 1):
            byte_pair: tuple[bytes, bytes] = (bytes_pretoken[left_idx], bytes_pretoken[left_idx + 1])
            bytes_pair_to_count[byte_pair] += ct
            bytes_pair_to_idxs[byte_pair].add(idx)

    return bytes_pair_to_count, bytes_pair_to_idxs

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
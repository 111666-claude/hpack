"""
Pure-bytes HPACK decoder hardening tests.

These tests exercise the decoder paths that matter when an untrusted peer
sends a malformed header block:

* integer continuation handling (RFC 7541 section 5.1),
* dynamic table size updates and their rollback on error,
* static/dynamic table index range checks,
* Huffman end-of-string padding and extreme Huffman inputs (section 5.2),
* literal Never-Indexed representation,
* the decompressed header list size budget,
* and the guarantee that *any* decoding error leaves the compression
  context exactly as it was before the rejected header block.

The suite only uses the standard library ``unittest`` runner and hard-coded
byte literals, so it can be executed standalone with::

    python -m unittest tests.test_hpack_decoder_hardening

or through ``pytest``.
"""

from __future__ import annotations

import unittest

from hpack import (
    Decoder,
    HeaderTuple,
    HPACKDecodingError,
    InvalidTableIndex,
    InvalidTableSizeError,
    NeverIndexedHeaderTuple,
    OversizedHeaderListError,
)
from hpack.hpack import decode_integer, encode_integer
from hpack.huffman_table import decode_huffman
from hpack.table import HeaderTable, table_entry_size

# A literal header field with incremental indexing: 0x40, one-byte name
# "a", one-byte value "b". Entry size is 32 + 1 + 1 = 34 octets.
LITERAL_AB = b"\x40\x01a\x01b"

# First valid dynamic table index (static table length + 1).
FIRST_DYNAMIC_INDEX = HeaderTable.STATIC_TABLE_LENGTH + 1

# A 7-bit-prefix integer that resolves to an index far beyond any populated
# table: prefix max (127) followed by a single terminating continuation
# byte, i.e. index 128.
UNREACHABLE_INDEX = b"\xff\x01"

# Hard-coded Huffman strings. "a" is the 5-bit code 00011 padded with three
# one-bits, giving the single byte 0x1f.
HUFFMAN_A = b"\x1f"


class IntegerContinuationTests(unittest.TestCase):
    """RFC 7541 section 5.1 variable-length integer decoding."""

    def test_rfc_1337_example(self) -> None:
        # The worked example from RFC 7541 section 5.1: 1337 with a 5-bit
        # prefix is encoded as 0x1f 0x9a 0x0a.
        self.assertEqual(decode_integer(b"\x1f\x9a\x0a", 5), (1337, 3))
        self.assertEqual(bytes(encode_integer(1337, 5)), b"\x1f\x9a\x0a")

    def test_continuation_shift_accumulates(self) -> None:
        # 7-bit prefix saturated at 127, then a terminating continuation of
        # value 1 with no shift: 127 + (1 << 0) = 128.
        self.assertEqual(decode_integer(b"\x7f\x01", 7), (128, 2))
        # A terminating zero continuation contributes nothing beyond the
        # saturated prefix: 127 + (0 << 0) = 127.
        self.assertEqual(decode_integer(b"\x7f\x00", 7), (127, 2))
        # An extra zero-valued (non-terminating) continuation shifts the
        # following terminator by an extra 7 bits but leaves the value at 255.
        self.assertEqual(decode_integer(b"\x7f\x80\x01", 7), (255, 3))
        # Two shifts: 127 + 0 + (1 << 14).
        self.assertEqual(decode_integer(b"\x7f\x80\x80\x01", 7), (16511, 4))

    def test_round_trip_across_prefixes_and_magnitudes(self) -> None:
        for prefix_bits in (4, 5, 6, 7, 8):
            for value in (
                0,
                2**prefix_bits - 1,
                2**prefix_bits,
                200,
                1337,
                2**14 - 1,
                2**31 - 1,
                2**32,
            ):
                encoded = bytes(encode_integer(value, prefix_bits))
                decoded, consumed = decode_integer(encoded, prefix_bits)
                self.assertEqual(decoded, value)
                self.assertEqual(consumed, len(encoded))

    def test_truncated_continuation_is_decoding_error(self) -> None:
        decoder = Decoder()
        # Every blob ends while an integer still expects a terminating byte.
        for blob in (
            b"\x3f",  # 5-bit prefix saturated, no continuation at all
            b"\x3f\x80",  #   ... one non-terminating continuation
            b"\x0f\x80",  # 4-bit prefix saturated, dangling continuation
            b"\xff\x80",  # 7-bit prefix saturated, dangling continuation
            b"\x7f",  # 7-bit prefix saturated, no continuation
        ):
            with self.subTest(blob=blob):
                with self.assertRaises(HPACKDecodingError):
                    decoder.decode(blob)

    def test_overlong_integer_is_decoding_error(self) -> None:
        decoder = Decoder()
        # Saturated prefix plus five non-terminating continuation bytes plus
        # a terminator: more octets than the implementation limit allows.
        blob = b"\xff" + b"\x80" * 5 + b"\x01"
        with self.assertRaises(HPACKDecodingError):
            decoder.decode(blob)


class HuffmanTailBitTests(unittest.TestCase):
    """RFC 7541 section 5.2 Huffman padding / EOS rules."""

    def test_valid_single_symbol_with_one_padding(self) -> None:
        self.assertEqual(decode_huffman(HUFFMAN_A), b"a")

    def test_eos_padding_pattern_rejected(self) -> None:
        # A byte of all one-bits is the start of the 30-bit EOS pattern and
        # can never be a valid padded symbol.
        for blob in (b"\xff", b"\xff\xff\xff\xff\xff\xff\xff\xff"):
            with self.subTest(blob=blob):
                with self.assertRaises(HPACKDecodingError):
                    decode_huffman(blob)

    def test_padding_must_be_all_ones(self) -> None:
        # 0x18 and 0x1e carry the "a" symbol bits but pad with zeroes / mixed
        # bits instead of the required run of one-bits.
        for blob in (b"\x18", b"\x1e"):
            with self.subTest(blob=blob):
                with self.assertRaises(HPACKDecodingError):
                    decode_huffman(blob)

    def test_incomplete_string_rejected(self) -> None:
        for blob in (b"\x80", b"\x00", b"\x5f\xff\xff\xff\xff"):
            with self.subTest(blob=blob):
                with self.assertRaises(HPACKDecodingError):
                    decode_huffman(blob)

    def test_empty_huffman_string(self) -> None:
        self.assertEqual(decode_huffman(b""), b"")

    def test_extreme_repeated_valid_symbol(self) -> None:
        # Huffman codes are not octet-aligned, so a repeated symbol forms a
        # periodic multi-byte stream rather than a repeated single byte. The
        # five-byte pattern below repeated 512 times (2560 bytes) decodes to
        # 4096 "a" symbols.
        size = 512
        encoded = b"\x18\xc6\x31\x8c\x63" * size
        self.assertEqual(decode_huffman(encoded), b"a" * (8 * size))

    def test_long_invalid_tail_poisons_whole_string(self) -> None:
        # A long valid stream followed by an invalid all-ones tail must still
        # be rejected rather than silently truncating the output.
        blob = b"\x18\xc6\x31\x8c\x63" * 64 + b"\xff"
        with self.assertRaises(HPACKDecodingError):
            decode_huffman(blob)

    def test_invalid_huffman_header_value_raises_and_rolls_back(self) -> None:
        decoder = Decoder()
        # Never-indexed literal, literal name "a", one-byte Huffman value
        # 0xff (invalid padding).
        with self.assertRaises(HPACKDecodingError):
            decoder.decode(b"\x10\x01a\x81\xff")
        self.assertEqual(len(decoder.header_table.dynamic_entries), 0)

    def test_invalid_huffman_header_name_raises_and_rolls_back(self) -> None:
        decoder = Decoder()
        # Incremental-indexing literal claiming a one-byte Huffman name 0xff.
        with self.assertRaises(HPACKDecodingError):
            decoder.decode(b"\x40\x81\xff\x01x")
        self.assertEqual(len(decoder.header_table.dynamic_entries), 0)


class TableIndexRangeTests(unittest.TestCase):
    """Static + dynamic table index validation."""

    def test_index_zero_is_invalid(self) -> None:
        with self.assertRaises(InvalidTableIndex):
            Decoder().decode(b"\x80")

    def test_last_static_entry_resolves(self) -> None:
        # Static table has 61 entries; index 61 is the final one.
        blob = bytes([0x80 | HeaderTable.STATIC_TABLE_LENGTH])
        self.assertEqual(
            Decoder().decode(blob),
            [("www-authenticate", "")],
        )

    def test_dynamic_index_with_empty_dynamic_table(self) -> None:
        # The first dynamic index cannot resolve before anything was added.
        blob = bytes([0x80 | FIRST_DYNAMIC_INDEX])
        with self.assertRaises(InvalidTableIndex):
            Decoder().decode(blob)

    def test_index_far_beyond_table(self) -> None:
        with self.assertRaises(InvalidTableIndex):
            Decoder().decode(UNREACHABLE_INDEX)

    def test_invalid_indexed_name_in_literal(self) -> None:
        # Literal without indexing whose 4-bit name index points beyond the
        # table. The prefix is saturated at 15 and a zero continuation plus a
        # terminating 1 pushes the name index to 143 (15 + 128), which no
        # static or dynamic entry occupies.
        with self.assertRaises(InvalidTableIndex):
            Decoder().decode(b"\x0f\x80\x01\x01x")


class NeverIndexedTests(unittest.TestCase):
    """Literal Never-Indexed representation (0x10 / 0x14 prefixes)."""

    def test_literal_name_never_indexed(self) -> None:
        decoder = Decoder()
        headers = decoder.decode(b"\x10\x01a\x01b")
        self.assertIsInstance(headers[0], NeverIndexedHeaderTuple)
        # Never-indexed fields must not enter the dynamic table.
        self.assertEqual(len(decoder.header_table.dynamic_entries), 0)

    def test_indexed_name_never_indexed(self) -> None:
        decoder = Decoder()
        # 0x14: never-indexed with name index 4 (":path"), empty value.
        headers = decoder.decode(b"\x14\x00", raw=True)
        self.assertIsInstance(headers[0], NeverIndexedHeaderTuple)
        self.assertEqual(tuple(headers[0]), (b":path", b""))
        self.assertEqual(len(decoder.header_table.dynamic_entries), 0)

    def test_plain_no_index_literal_is_not_never_indexed(self) -> None:
        decoder = Decoder()
        headers = decoder.decode(b"\x00\x01z\x00")
        self.assertIs(type(headers[0]), HeaderTuple)
        self.assertEqual(len(decoder.header_table.dynamic_entries), 0)


class RepeatedIndexTests(unittest.TestCase):
    """Repeated references into the dynamic table (compression amplification)."""

    def test_repeated_dynamic_index_references(self) -> None:
        decoder = Decoder()
        reference = bytes([0x80 | FIRST_DYNAMIC_INDEX])
        repetitions = 128
        block = LITERAL_AB + reference * repetitions
        headers = decoder.decode(block, raw=True)
        self.assertEqual(len(headers), 1 + repetitions)
        self.assertEqual(tuple(headers[0]), (b"a", b"b"))
        self.assertTrue(
            all(tuple(header) == (b"a", b"b") for header in headers[1:]),
        )
        # Referencing an entry never duplicates it in the table.
        self.assertEqual(len(decoder.header_table.dynamic_entries), 1)

    def test_repeated_static_index_references(self) -> None:
        decoder = Decoder()
        # Index 2 is ":method: GET" from the static table.
        headers = decoder.decode(b"\x82" * 256)
        self.assertEqual(len(headers), 256)
        self.assertTrue(all(header == (":method", "GET") for header in headers))
        self.assertEqual(len(decoder.header_table.dynamic_entries), 0)


class HeaderListBudgetTests(unittest.TestCase):
    """Decompressed header list size enforcement and rollback."""

    def test_entry_exactly_at_budget_accepted(self) -> None:
        entry_size = table_entry_size(b"a", b"b")
        self.assertEqual(entry_size, 34)
        decoder = Decoder(max_header_list_size=entry_size)
        self.assertEqual(decoder.decode(LITERAL_AB), [("a", "b")])

    def test_entry_one_over_budget_rejected(self) -> None:
        decoder = Decoder(max_header_list_size=table_entry_size(b"a", b"b") - 1)
        with self.assertRaises(OversizedHeaderListError):
            decoder.decode(LITERAL_AB)

    def test_cumulative_budget_enforced(self) -> None:
        # Two 34-octet entries against a 40-octet budget: the first fits, the
        # second pushes the running total over.
        decoder = Decoder(max_header_list_size=40)
        with self.assertRaises(OversizedHeaderListError):
            decoder.decode(LITERAL_AB * 2)

    def test_budget_failure_rolls_back_dynamic_table(self) -> None:
        decoder = Decoder(max_header_list_size=40)
        with self.assertRaises(OversizedHeaderListError):
            decoder.decode(LITERAL_AB * 2)
        self.assertEqual(len(decoder.header_table.dynamic_entries), 0)
        self.assertEqual(decoder.header_table._current_size, 0)

    def test_decoder_reusable_after_budget_failure(self) -> None:
        decoder = Decoder(max_header_list_size=40)
        with self.assertRaises(OversizedHeaderListError):
            decoder.decode(LITERAL_AB * 2)
        # Index 4 is ":path: /" (entry size 38), which fits the same budget
        # and must still decode cleanly after the rejected block.
        self.assertEqual(decoder.decode(b"\x84"), [(":path", "/")])
        self.assertEqual(len(decoder.header_table.dynamic_entries), 0)


class ContextRollbackTests(unittest.TestCase):
    """Every decoding error must restore the pre-call compression context."""

    def _primed_decoder(self) -> Decoder:
        decoder = Decoder()
        decoder.decode(LITERAL_AB)
        return decoder

    def test_literal_indexing_rolled_back_after_invalid_index(self) -> None:
        decoder = self._primed_decoder()
        snapshot = list(decoder.header_table.dynamic_entries)
        with self.assertRaises(InvalidTableIndex):
            decoder.decode(b"\x40\x01c\x01d" + UNREACHABLE_INDEX)
        self.assertEqual(list(decoder.header_table.dynamic_entries), snapshot)
        self.assertEqual(
            decoder.header_table._current_size,
            table_entry_size(b"a", b"b"),
        )

    def test_truncated_literal_rolls_back(self) -> None:
        decoder = self._primed_decoder()
        snapshot = list(decoder.header_table.dynamic_entries)
        # A valid literal followed by one claiming a five-byte name it does
        # not actually contain.
        with self.assertRaises(HPACKDecodingError):
            decoder.decode(b"\x40\x01c\x01d" + b"\x40\x05ab")
        self.assertEqual(list(decoder.header_table.dynamic_entries), snapshot)

    def test_invalid_huffman_rolls_back_prior_literal(self) -> None:
        decoder = self._primed_decoder()
        snapshot = list(decoder.header_table.dynamic_entries)
        with self.assertRaises(HPACKDecodingError):
            decoder.decode(b"\x40\x01c\x01d" + b"\x10\x01a\x81\xff")
        self.assertEqual(list(decoder.header_table.dynamic_entries), snapshot)

    def test_table_size_update_rolled_back_after_later_error(self) -> None:
        decoder = self._primed_decoder()
        max_size = decoder.header_table.maxsize
        entries = list(decoder.header_table.dynamic_entries)
        # Shrink the table to zero (0x20) and then reference an invalid index.
        with self.assertRaises(HPACKDecodingError):
            decoder.decode(b"\x20" + UNREACHABLE_INDEX)
        self.assertEqual(decoder.header_table.maxsize, max_size)
        self.assertEqual(list(decoder.header_table.dynamic_entries), entries)

    def test_multiple_size_updates_rolled_back(self) -> None:
        decoder = Decoder()
        max_size = decoder.header_table.maxsize
        # Two valid size updates (to 10, then to 21) followed by a bad index.
        with self.assertRaises(HPACKDecodingError):
            decoder.decode(b"\x2a\x35" + UNREACHABLE_INDEX)
        self.assertEqual(decoder.header_table.maxsize, max_size)

    def test_oversize_table_update_rolled_back(self) -> None:
        decoder = Decoder()
        decoder.max_allowed_table_size = 100
        with self.assertRaises(InvalidTableSizeError):
            # 5-bit saturated prefix + continuation: 31 + 102 = 133 > 100.
            decoder.decode(b"\x3f\x66")
        self.assertEqual(decoder.header_table.maxsize, HeaderTable.DEFAULT_SIZE)
        self.assertEqual(len(decoder.header_table.dynamic_entries), 0)

    def test_successful_block_still_updates_context(self) -> None:
        # Rollback must not affect the normal success path.
        decoder = Decoder()
        decoder.decode(LITERAL_AB)
        self.assertEqual(
            list(decoder.header_table.dynamic_entries),
            [(b"a", b"b")],
        )
        decoder.max_allowed_table_size = 100
        decoder.decode(b"\x2a\x82")  # shrink to 10, then ":method GET".
        self.assertEqual(decoder.header_table.maxsize, 10)


if __name__ == "__main__":
    unittest.main()

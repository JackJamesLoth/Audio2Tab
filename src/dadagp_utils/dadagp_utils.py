"""Small public wrapper around the bundled DadaGP conversion code."""

import guitarpro as gp

from .dadagp import dadagp_decode, guitarpro2tokens


class DadaGP:
    """Encode Guitar Pro files as DadaGP tokens and decode tokens to GP5."""

    def __init__(self, filename, artist_token="unknown", roundTempo=False):
        song = gp.parse(filename)
        self.__text_encoding = guitarpro2tokens(
            song,
            artist_token,
            roundTempo,
            verbose=False,
        )

    def get_text_encoding(self):
        return self.__text_encoding

    @staticmethod
    def decode(input_file, output_file):
        dadagp_decode(input_file, output_file)

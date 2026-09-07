"""The dual-layout quantizer's transposed operand, stored in B-operand order."""

import pytest
import torch

from lumen.ops.quantize.linear import _shuffle_mxfp4_weight
from lumen.ops.quantize.ops import dual_layout_quant_mxfp4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

G = 16
BLOCK = 32


@pytest.mark.parametrize("shape", [(256, 128), (512, 256), (1024, 512)])
def test_shuffled_col_operand_matches_separate_shuffle_pass(shape):
    M, N = shape
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    sign = (torch.randint(0, 2, (G,), device="cuda") * 2 - 1).to(torch.bfloat16)

    row, row_s, col, col_s = dual_layout_quant_mxfp4(
        x, sign, block_size=BLOCK, g=G, use_sr_row=False, use_sr_transposed=False,
    )
    row_f, row_sf, col_f, col_sf = dual_layout_quant_mxfp4(
        x, sign, block_size=BLOCK, g=G, use_sr_row=False, use_sr_transposed=False,
        shuffle_col=True,
    )

    # Only the transposed operand's data layout changes.
    torch.testing.assert_close(row_f, row, rtol=0, atol=0)
    torch.testing.assert_close(row_sf, row_s, rtol=0, atol=0)
    torch.testing.assert_close(col_sf, col_s, rtol=0, atol=0)
    torch.testing.assert_close(col_f, _shuffle_mxfp4_weight(col), rtol=0, atol=0)

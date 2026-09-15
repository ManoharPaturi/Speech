# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch


@pytest.mark.unit
def test_magpie_tts_padding_logic_exact_multiple():
    """Verify that audio with length that is an exact multiple of codec_model_samples_per_frame
    is not padded with a spurious extra frame."""
    codec_model_samples_per_frame = 256
    audio_exact = torch.zeros(codec_model_samples_per_frame * 4)
    remainder = audio_exact.shape[0] % codec_model_samples_per_frame
    if remainder != 0:
        padded = torch.nn.functional.pad(
            audio_exact,
            (0, codec_model_samples_per_frame - remainder),
            value=0,
        )
    else:
        padded = audio_exact

    assert padded.shape[0] == codec_model_samples_per_frame * 4


@pytest.mark.unit
def test_magpie_tts_padding_logic_non_multiple():
    """Verify that audio with length that is not an exact multiple is padded to the next multiple."""
    codec_model_samples_per_frame = 256
    audio_non_exact = torch.zeros(codec_model_samples_per_frame * 4 - 10)
    remainder = audio_non_exact.shape[0] % codec_model_samples_per_frame
    assert remainder != 0
    padded = torch.nn.functional.pad(
        audio_non_exact,
        (0, codec_model_samples_per_frame - remainder),
        value=0,
    )
    assert padded.shape[0] == codec_model_samples_per_frame * 4

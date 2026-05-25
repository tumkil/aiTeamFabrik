# Copyright 2024 SoftwareTeamFabrik contributors
# SPDX-License-Identifier: MIT

# Shared constants for the factory.core module

# Sentinel prefix used for machine-generated Architect Analysis notes.
# Matching is done via startswith() so human comments that merely mention
# "Architect Analysis" in the middle of a sentence are not affected.
ARCH_NOTE_PREFIX = "## Architect Analysis\n\n"

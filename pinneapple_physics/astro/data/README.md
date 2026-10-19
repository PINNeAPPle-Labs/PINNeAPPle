# Data shipped with `pinneapple_physics.astro`

Built by `scripts/astro/build_catalogs.py`. All sources are open; nothing here comes from GPL projects.

| File | Content | Source and licence |
|---|---|---|
| `bsc5.json.gz` | 9,096 stars to V ~ 6.5: J2000 position, proper motion, V, B-V, spectral type, parallax, radial velocity | *Yale Bright Star Catalogue*, 5th revised ed., D. Hoffleit & W. H. Warren Jr. (1991), NASA Astronomical Data Center / CDS catalogue V/50 (freely distributed for research and education). Read from the JSON conversion in github.com/brettonw/YaleBrightStarCatalog (MIT, (c) 2016 Bretton Wade). Proper names matched by HD number from d3-celestial `starnames.json` (below). |
| `constellations.json.gz` | IAU constellation names, label points and stick figures (J2000 RA/Dec) | d3-celestial data, (c) 2015 Olaf Frohn, BSD-3-Clause (licence text below). |
| `constellation_names.json` | Constellation names by IAU abbreviation (Serpens as `Ser1` Caput / `Ser2` Cauda): Latin (IAU) and Brazilian Portuguese | IAU list (1922); Portuguese names in common Brazilian use (R. R. de Freitas Mourão, *Dicionário Enciclopédico de Astronomia e Astronáutica*, 1987). Written for this project. |
| `facts/` | Info cards (Learn mode): Solar System, brightest stars, constellations, Messier objects, satellites, with a source per number and text in English and Portuguese | See `facts/README.md` (NASA fact sheets, Hipparcos 2007, IAU, d3-celestial BSD-3, published distances). Text written for this project. |

The planetary ephemeris is **not** stored here: JPL DE421 (public domain, JPL/NASA) is read from the `skyfield-data`
package (MIT) or from `$PINNEAPPLE_DE_KERNEL`.

## d3-celestial licence

Copyright (c) 2015, Olaf Frohn
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
   disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
   following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
   products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Yale BSC JSON conversion licence (MIT)

Copyright (c) 2016 Bretton Wade. Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files, to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies, subject to
the following condition: the above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

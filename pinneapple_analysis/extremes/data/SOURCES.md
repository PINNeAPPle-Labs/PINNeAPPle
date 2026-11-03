# Bundled extreme-value data

Public observational series used by Coles, *An Introduction to Statistical Modeling of Extreme Values*
(Springer, 2001), as distributed with the R package `ismev` (data folder, files `portpirie.rda`, `fremantle.rda`,
`rain.rda`). They are measurements, converted to CSV without any change of value.

| File | Content | Used in Coles (2001) |
|---|---|---|
| `portpirie.csv` | annual maximum sea level (m) at Port Pirie, South Australia, 1923-1987 (65 years) | §3.4.1, stationary GEV |
| `fremantle.csv` | annual maximum sea level (m) at Fremantle, Western Australia, 1897-1989 (86 years with data), and the annual mean Southern Oscillation Index | §6.3.1, GEV with a linear trend and the SOI in the location |
| `rain.csv` | daily rainfall (mm) at a location in south-west England, 1914-1962 (17 531 days) | §4.4.1, GPD above 30 mm |
